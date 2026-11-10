"""Check CPU dependencies and a small Keras -> LiteRT conversion, without datasets."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

PINNED = {"tensorflow": "2.21.0", "keras": "3.13.2", "ai-edge-litert": "2.3.0"}
MODULES = {
    "tensorflow": "tensorflow", "keras": "keras", "ai-edge-litert": "ai_edge_litert",
    "numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
    "matplotlib": "matplotlib", "psutil": "psutil", "openpyxl": "openpyxl",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write local environment information as JSON")
    args = parser.parse_args()
    errors = []
    versions = {}
    if sys.prefix == sys.base_prefix:
        errors.append("가상환경의 Python으로 실행해주세요.")
    if sys.version_info[:2] != (3, 12):
        errors.append("공통 환경 후보는 Python 3.12입니다.")
    for package in MODULES:
        try:
            versions[package] = importlib.metadata.version(package)
            if package in PINNED and versions[package] != PINNED[package]:
                errors.append(f"{package}: {PINNED[package]} 필요, 현재 {versions[package]}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package}: 미설치")
    report = {
        "python": platform.python_version(), "os": platform.system(),
        "os_release": platform.release(), "macos_version": platform.mac_ver()[0] or None,
        "architecture": platform.machine(),
        "packages": versions, "status": "incomplete", "smoke_check": "not_run",
        "errors": errors,
    }
    if not errors:
        # Put writable caches inside this virtual environment, not global user configuration.
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("KERAS_HOME", str(Path(sys.prefix) / ".keras"))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".matplotlib"))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            for module in MODULES.values():
                importlib.import_module(module)
            import numpy as np
            import tensorflow as tf
            import keras
            from ai_edge_litert.interpreter import Interpreter
            tf.config.set_visible_devices([], "GPU")
            if keras.backend.backend() != "tensorflow":
                raise RuntimeError("Keras backend must be TensorFlow")
            model = keras.Sequential([
                keras.layers.Input(shape=(3,)),
                keras.layers.Dense(2, use_bias=False, kernel_initializer="ones"),
            ])
            sample = np.ones((1, 3), dtype=np.float32)
            reference = model(sample, training=False).numpy()
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converted = converter.convert()
            interpreter = Interpreter(model_content=converted, num_threads=1)
            interpreter.allocate_tensors()
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], sample)
            interpreter.invoke()
            actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
            report.update(status="passed", smoke_check="keras_to_tflite_fp32_and_litert_cpu", output_shape=list(actual.shape))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            report["status"] = "failed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
