"""
ONNX Runtime inference session wrapper.
Supports CUDAExecutionProvider, TensorrtExecutionProvider, CPUExecutionProvider.
"""

import os
import numpy as np
import onnxruntime as ort
from typing import List

# ImageNet normalisation constants (same as training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

PROVIDER_MAP = {
    "cpu":       ["CPUExecutionProvider"],
    "cuda":      ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt":  ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def _build_session(model_path: str, provider: str) -> ort.InferenceSession:
    providers = PROVIDER_MAP.get(provider, PROVIDER_MAP["cpu"])

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count()

    trt_options = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/tmp/trt_cache",
    }

    provider_options = []
    for p in providers:
        if p == "TensorrtExecutionProvider":
            provider_options.append(trt_options)
        else:
            provider_options.append({})

    session = ort.InferenceSession(model_path, sess_options=opts,
                                   providers=providers,
                                   provider_options=provider_options)
    return session


class OnnxInferenceSession:
    def __init__(self, model_path: str, provider: str = "cpu", class_names: List[str] = None):
        self.session = _build_session(model_path, provider)
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.class_names = class_names or [str(i) for i in range(196)]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Expects HxWx3 uint8 numpy array, returns 1x3x224x224 float32."""
        from PIL import Image
        img = Image.fromarray(image).resize((224, 224))
        x = np.array(img, dtype=np.float32) / 255.0          # H W C
        x = x.transpose(2, 0, 1)[np.newaxis]                  # 1 C H W
        x = (x - _MEAN) / _STD
        return x.astype(np.float32)

    def predict(self, image: np.ndarray) -> dict:
        x = self.preprocess(image)
        (logits,) = self.session.run(None, {self.input_name: x})
        probs = _softmax(logits[0])
        top5_idx = probs.argsort()[::-1][:5]
        return {
            "class_id":   int(top5_idx[0]),
            "class_name": self.class_names[top5_idx[0]],
            "confidence": float(probs[top5_idx[0]]),
            "top5": [
                {"class_id": int(i), "class_name": self.class_names[i], "confidence": float(probs[i])}
                for i in top5_idx
            ],
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
