"""A lossless frame cache: shared mmap arrays and the original encoded images.

The cache avoids reparsing an entire parquet episode for each shuffled frame.
It preserves episode boundaries, masks, raw taxels, image bytes and calibration.
Source size/mtime fingerprints are checked before a cache can be used.
"""
import bisect
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from openpi.models.tactile_contact import calibrate_baseline, contact_fields
from openpi.vtla.pressure import calibrate_o10, o10_contact_fields
from openpi.training.data_loader import LocalLeRobotParquetDataset

CAMERAS = ('top', 'left_wrist', 'right_wrist')


def build_frame_cache(root, *, baseline_frames=16):
    import pyarrow.parquet as pq
    root = Path(root)
    destination = root / 'frame_cache'
    with (root / '.frame_cache.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            validate_cache(root)
            return destination
        episodes = [json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()]
        lengths = [int(e.get('length', e.get('n_frames'))) for e in episodes]
        ends = np.cumsum(lengths)
        n, count = int(ends[-1]), len(episodes)
        info = json.loads((root/'meta/info.json').read_text())
        native = info.get('tactile_layout') == 'o10_268'
        hands = ('right',) if native else ('left','right')
        shape = (1,1,268) if native else (11,14,14)
        sensors = shape[0]
        with tempfile.TemporaryDirectory(prefix='.frame_cache_build_', dir=root) as temporary:
            out = Path(temporary)
            specs = {'state': ((n, info['features']['observation.state']['shape'][0]), 'float32'), 'actions': ((n, info['features']['action']['shape'][0]), 'float32'),
                     'task_index': ((n,), 'int64'), 'timestamp': ((n,), 'float64')}
            for hand in hands:
                specs.update({f'tactile_{hand}': ((n,*shape), 'float32'),
                              f'valid_{hand}': ((n,sensors), 'bool'),
                              f'baseline_{hand}': ((count,*shape), 'float32'),
                              f'scale_{hand}': ((count,sensors), 'float32'),
                              f'baseline_valid_{hand}': ((count,sensors), 'bool')})
            for camera in CAMERAS:
                specs[f'offsets_{camera}'] = ((n+1,), 'int64')
            arrays = {k: np.lib.format.open_memmap(out/f'{k}.npy', mode='w+', dtype=d, shape=s)
                      for k, (s,d) in specs.items()}
            streams = {c: (out/f'images_{c}.bin').open('wb') for c in CAMERAS}
            sources = []
            try:
                start = 0
                for pos, (episode, length) in enumerate(zip(episodes, lengths, strict=True)):
                    ep = int(episode['episode_index'])
                    files = list((root/'data').glob(f'chunk-*/episode_{ep:06d}.parquet'))
                    if len(files) != 1:
                        raise ValueError(f'Expected one parquet for episode {ep}')
                    source = files[0]
                    before = source.stat()
                    df = pq.read_table(source).to_pandas()
                    if len(df) != length:
                        raise ValueError(f'Length mismatch in episode {ep}')
                    stop = start + length
                    for key, column in (('state','observation.state'), ('actions','action')):
                        values = np.stack(df[column]).astype(np.float32)
                        if not np.isfinite(values).all():
                            raise ValueError(f'Nonfinite {column} in episode {ep}')
                        arrays[key][start:stop] = values
                    arrays['task_index'][start:stop] = np.asarray(df['task_index'], np.int64)
                    arrays['timestamp'][start:stop] = np.asarray(df['timestamp'], np.float64)
                    for hand in hands:
                        touch = np.stack(df[f'observation.tactile.{hand}']).astype(np.float32).reshape(length,*shape)
                        valid = np.stack(df[f'observation.tactile.{hand}_valid']).astype(bool)
                        arrays[f'tactile_{hand}'][start:stop] = touch
                        arrays[f'valid_{hand}'][start:stop] = valid
                        baseline, scale, usable = (calibrate_o10 if native else calibrate_baseline)(touch[:baseline_frames], valid[:baseline_frames])
                        arrays[f'baseline_{hand}'][pos] = baseline
                        arrays[f'scale_{hand}'][pos] = scale
                        arrays[f'baseline_valid_{hand}'][pos] = usable
                    for camera in CAMERAS:
                        stream, offsets = streams[camera], arrays[f'offsets_{camera}']
                        for frame, value in enumerate(df[f'observation.images.{camera}']):
                            offsets[start+frame] = stream.tell()
                            stream.write(value['bytes'])
                        offsets[stop] = stream.tell()
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError('Parquet changed during cache creation')
                    sources.append({'path':str(source.relative_to(root)), 'bytes':after.st_size,
                                    'mtime_ns':after.st_mtime_ns})
                    start = stop
                    print(f'Frame cache: {pos+1}/{count} episodes, {stop}/{n} frames', flush=True)
                for values in arrays.values():
                    values.flush()
            finally:
                for stream in streams.values():
                    stream.close()
            manifest = {'version':1, 'tactile_layout':info.get('tactile_layout','inspire11'), 'total_frames':n, 'baseline_frames':baseline_frames,
                        'episode_ids':[int(e['episode_index']) for e in episodes],
                        'episode_ends':ends.tolist(), 'sources':sources, 'arrays':list(specs)}
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            os.rename(out, destination)
    return destination


def validate_cache(root):
    root = Path(root)
    manifest = json.loads((root/'frame_cache/manifest.json').read_text())
    if manifest['version'] != 1:
        raise ValueError('Unsupported frame cache version')
    for source in manifest['sources']:
        st = (root/source['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
            raise ValueError(f'Stale frame cache for {source["path"]}; rebuild explicitly')
    return manifest


class CachedLeRobotDataset(LocalLeRobotParquetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest = validate_cache(self._root)
        if self.manifest['baseline_frames'] != self._baseline_frames:
            raise ValueError('Cache calibration interval differs from training')
        if self.manifest['episode_ends'] != self._episode_ends:
            raise ValueError('Cache episode boundaries differ from metadata')
        if self.manifest['episode_ids'] != [int(e['episode_index']) for e in self._episodes]:
            raise ValueError('Cache episode IDs differ from metadata')
        self._hand = self._tactile_key.rsplit('.',1)[-1]
        self._arrays = None
        self._pid = None
        self._image_fds = {}

    def _open(self):
        if self._pid == os.getpid():
            return
        for fd in self._image_fds.values():
            os.close(fd)
        self._pid = os.getpid()
        cache = self._root/'frame_cache'
        self._arrays = {name:np.load(cache/f'{name}.npy',mmap_mode='r') for name in self.manifest['arrays']}
        self._image_fds = {camera:os.open(cache/f'images_{camera}.bin', os.O_RDONLY) for camera in CAMERAS}

    def _tactile_window(self, unused, local_idx):
        self._open()
        n = int(self._tactile_num_frames or 16)
        first = max(0, local_idx-n+1)
        a, b = self._episode_start+first, self._episode_start+local_idx+1
        native = self.manifest.get('tactile_layout') == 'o10_268'
        shape = (1,n,1,268) if native else (11,n,14,14)
        taxels = np.zeros(shape,np.float32)
        valid = np.zeros(shape[:2],bool)
        length = b-a
        taxels[:,-length:] = self._arrays[f'tactile_{self._hand}'][a:b].transpose(1,0,2,3)
        valid[:,-length:] = self._arrays[f'valid_{self._hand}'][a:b].T
        if self._contact_supervision:
            return (o10_contact_fields if native else contact_fields)(taxels, valid, *self._tactile_calibration,
                                  calibrated=local_idx >= self._baseline_frames-1,
                                  off_threshold=self._contact_off, on_threshold=self._contact_on)
        taxels *= valid[:,:,None,None]
        frame_valid = np.zeros(n,bool)
        frame_valid[-length:] = True
        return {'tactile':taxels, 'tactile_valid':valid.any(axis=1), 'tactile_frame_valid':frame_valid}

    def __getitem__(self, index):
        self._open()
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pos = bisect.bisect_right(self._episode_ends,index)
        self._episode_start = 0 if pos == 0 else self._episode_ends[pos-1]
        end = self._episode_ends[pos]
        local = index-self._episode_start
        self._tactile_calibration = tuple(self._arrays[f'{field}_{self._hand}'][pos]
                                          for field in ('baseline','scale','baseline_valid'))
        action_indices = np.minimum(np.arange(index,index+self._action_horizon),end-1)
        sample = {'observation.state':np.array(self._arrays['state'][index]),
                  'action':np.array(self._arrays['actions'][action_indices]),
                  'task_index':np.asarray(self._arrays['task_index'][index],np.int64),
                  'sample_global_index':np.asarray(index,np.int64),
                  'sample_episode_index':np.asarray(self._episodes[pos]['episode_index'],np.int64),
                  'sample_frame_in_episode':np.asarray(local,np.int64)}
        for camera in CAMERAS:
            offsets = self._arrays[f'offsets_{camera}']
            begin, stop = map(int,offsets[index:index+2])
            data = os.pread(self._image_fds[camera],stop-begin,begin)
            with Image.open(io.BytesIO(data)) as image:
                sample[f'observation.images.{camera}'] = np.asarray(image.convert('RGB'))
        if self._tactile_num_frames is not None:
            sample.update(self._tactile_window(None,local))
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_arrays=None,_image_fds={},_pid=None)
        return state
