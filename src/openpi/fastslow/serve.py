"""JSON service for pi05 fast/slow inference; no robot control calls."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import time

import numpy as np
from PIL import Image

from openpi.fastslow.policy import load_policy

DEFAULT_CHECKPOINT = "/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla_full/full_from_10000_30k/29999"


def make_server(policy, host="127.0.0.1", port=18771):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": "Unknown route"})
            self.reply(200, {"status": "ready", "model": "pi05", "session_id": policy.session_id,
                             "chunk_id": policy.chunk_id, "slow_steps": policy.model.slow_steps,
                             "fast_steps": policy.model.flow_steps-policy.model.slow_steps,
                             "state_refresh": "slow", "robot_control": False})

        def do_POST(self):
            started = time.monotonic()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16*1024*1024:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                if self.path == "/reset":
                    result = {"session_id": policy.reset()}
                elif self.path == "/pressure":
                    policy.feed_pressure(data["session_id"], data["raw"], data["valid"], data["timestamp"])
                    result = {"calibrated": policy.stream.calibration is not None}
                elif self.path == "/slow":
                    images = {}
                    for key, value in data["images"].items():
                        with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as im:
                            images[key] = np.asarray(im.convert("RGB"))
                    result = policy.slow(data["session_id"], {"images": images, "state": data["state"],
                                         "prompt": data["prompt"]}, data["timestamp"])
                elif self.path == "/fast":
                    result = policy.fast(data["session_id"], data["chunk_id"], data["timestamp"],
                                         data["executed_steps"])
                else:
                    return self.reply(404, {"error": "Unknown route"})
                result["latency_ms"] = (time.monotonic()-started)*1000
                self.reply(200, result)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.reply(400, {"error": str(error)})
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.reply(500, {"error": type(error).__name__, "detail": str(error)})

    return ThreadingHTTPServer((host, port), Handler)


def warmup(policy):
    session = policy.reset()
    for i in range(16):
        policy.feed_pressure(session, np.zeros((11, 14, 14)), np.ones(11, bool), i/policy.fps)
    stamp = 15/policy.fps
    obs = {"images": {k: np.zeros((224, 224, 3), np.uint8) for k in ("top", "left_wrist", "right_wrist")},
           "state": np.zeros(26, np.float32), "prompt": "warmup"}
    slow = policy.slow(session, obs, stamp)
    policy.fast(session, slow["chunk_id"], stamp)
    policy.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="pi05_righthand_ee_fastslow")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18771)
    parser.add_argument("--fps", type=float, help="Optional check against checkpoint fps; cannot change action timing")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()
    policy = load_policy(args.checkpoint, args.config, fps=args.fps)
    if not args.skip_warmup:
        print("Compiling slow and fast inference before accepting live data...", flush=True)
        warmup(policy)
    server = make_server(policy, args.host, args.port)
    print(f"pi05 fast/slow ready: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
