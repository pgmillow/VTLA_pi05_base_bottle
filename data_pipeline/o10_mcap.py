"""Native O10/7-axis-arm MCAP reader; no fake left arm or Inspire patches."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

path = Path(__file__).parent/'canonical/mcap_to_lerobot_v21.py'
spec = importlib.util.spec_from_file_location('o10_v21_utils',path)
v21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v21
spec.loader.exec_module(v21)

HAND_NAMES = ['thumb_roll','thumb_abad','thumb_mcp','index_abad','index_pip',
              'middle_pip','ring_abad','ring_pip','pinky_abad','pinky_pip']
ARM_NAMES = [f'joint{i}' for i in range(1,8)]
NUMERIC = {'arm_q':('/right_arm/states',7),
           'arm_cmd':('/right_arm/desired_joint_target',7),
           'ree_p':('/right_arm/ee_position',3),
           'ree_r':('/right_arm/ee_rpy',3),
           'rhq':('/o10/right/joint_states',10),
           'rhc':('/o10/right/joint_cmd',10)}
IMAGES = {'top':v21.TOPIC_TOP,'left_img':v21.TOPIC_LEFT_IMG,'right_img':v21.TOPIC_RIGHT_IMG}
TACTILE_TOPIC = '/o10/right/tactile_states'


def decode_tactile(message):
    if (message.encoding, message.height, message.width) != ('mono16',1,268):
        raise ValueError('Expected O10 mono16 image with 1 x 268 raw channels')
    raw = bytes(message.data)
    if message.step < 536 or len(raw) < message.step:
        raise ValueError('Truncated tactile packet')
    dtype = '>u2' if message.is_bigendian else '<u2'
    return np.frombuffer(raw[:536],dtype=dtype).astype(np.float32)


def load_mcap(path):
    numeric = {key:v21.NumBuf() for key in NUMERIC}
    images = {key:v21.ImgBuf() for key in IMAGES}
    touch = v21.NumBuf()
    by_topic = {topic:(key,dim) for key,(topic,dim) in NUMERIC.items()}
    image_by_topic = {topic:key for key,topic in IMAGES.items()}
    with Path(path).open('rb') as source:
        reader = make_reader(source,decoder_factories=[DecoderFactory()])
        for _,channel,record,message in reader.iter_decoded_messages(topics=[*by_topic,*image_by_topic,TACTILE_TOPIC]):
            stamp = v21._require_stamp(message,channel.topic,record.log_time)
            if channel.topic in image_by_topic:
                buffer = images[image_by_topic[channel.topic]]
                buffer.t.append(stamp); buffer.raw.append(message)
            elif channel.topic == TACTILE_TOPIC:
                touch.t.append(stamp); touch.v.append(decode_tactile(message))
            else:
                key,dim = by_topic[channel.topic]
                values = v21._vec3(message) if key in ('ree_p','ree_r') else np.asarray(message.position,np.float32)
                if values.shape != (dim,):
                    raise ValueError(f'{channel.topic}: expected {dim} values, got {values.shape}')
                names = list(getattr(message,'name',[]) or [])
                expected = (ARM_NAMES if key.startswith('arm_') else
                            ['x','y','z'] if key == 'ree_p' else
                            ['roll','pitch','yaw'] if key == 'ree_r' else HAND_NAMES)
                if names and names != expected:
                    raise ValueError(f'Unexpected joint order on {channel.topic}: {names}')
                buffer = numeric[key]
                buffer.t.append(stamp); buffer.v.append(values); buffer.names = names or expected
    # Stable sorting preserves log order within repeated header timestamps;
    # conversion can then deterministically retain the last published value.
    def numeric_sorted(buffer):
        order = np.argsort(buffer.t,kind='stable')
        return np.asarray(buffer.t,dtype=np.float64)[order], np.asarray(buffer.v,dtype=np.float32)[order], buffer.names
    result = {key:numeric_sorted(value) for key,value in numeric.items()}
    result.update({key:v21._sanitize_img(value) for key,value in images.items()})
    result['tactile'] = numeric_sorted(touch)
    missing = [key for key,value in result.items() if len(value[0]) == 0]
    if missing:
        raise ValueError('Required O10 streams are missing: '+', '.join(missing))
    return result
