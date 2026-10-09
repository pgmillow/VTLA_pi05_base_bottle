"""Robot-independent JSON client. Feed all sensor frames, not only fast ticks."""
import base64
import io
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .buffer import ActionUpdate


class Pi05FastSlowClient:
    def __init__(self, url="http://127.0.0.1:18771", timeout=10):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.session_id = None

    def _post(self, route, data):
        request = Request(self.url+route, json.dumps(data, allow_nan=False).encode(),
                          {"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read().decode()
            raise RuntimeError(f"pi05 {route}: HTTP {error.code}: {detail}") from error

    def reset(self):
        self.session_id = self._post("/reset", {})["session_id"]
        return self.session_id

    def feed_pressure(self, raw, valid, timestamp):
        if self.session_id is None:
            raise RuntimeError("Call reset before feeding pressure")
        raw = np.asarray(raw, np.float32).reshape(11, 14, 14)
        valid = np.asarray(valid, bool).reshape(11)
        # JSON has no NaN. Preserve missingness when replacing invalid values.
        from openpi.models.tactile_contact import spatial_mask
        valid = valid & np.isfinite(np.where(spatial_mask(), raw, 0)).all(axis=(-2, -1))
        raw = np.where(valid[:, None, None] & spatial_mask(), raw, 0)
        return self._post("/pressure", {"session_id": self.session_id, "raw": raw.tolist(),
            "valid": valid.tolist(), "timestamp": float(timestamp)})

    def slow(self, images, prompt, state, timestamp):
        encoded = {}
        for key, arr in images.items():
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError("Camera images must be RGB uint8 HWC")
            buffer = io.BytesIO()
            Image.fromarray(arr).save(buffer, format="PNG")
            encoded[key] = base64.b64encode(buffer.getvalue()).decode()
        return self._post("/slow", {"session_id": self.session_id, "images": encoded,
            "prompt": prompt, "state": np.asarray(state).tolist(), "timestamp": float(timestamp)})

    def fast(self, chunk_id, timestamp, executed_steps):
        result = self._post("/fast", {"session_id": self.session_id, "chunk_id": int(chunk_id),
            "timestamp": float(timestamp), "executed_steps": int(executed_steps)})
        keys = ActionUpdate.__dataclass_fields__
        values = {k: result[k] for k in keys}
        values["actions"] = np.asarray(values["actions"], np.float32)
        return ActionUpdate(**values), {k: v for k, v in result.items() if k not in keys}
