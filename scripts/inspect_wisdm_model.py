"""Inspect archive metadata without importing Keras or loading model weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def inspect():
    manifest = json.loads((ROOT / "configs/wisdm-model.json").read_text(encoding="utf-8"))
    path = ROOT / manifest["local_path"]
    if not path.is_file():
        raise ValueError(f"모델 파일이 없습니다: {manifest['local_path']}. docs/wisdm-model.md를 확인해주세요.")
    with path.open("rb") as model_file:
        actual_hash = hashlib.file_digest(model_file, "sha256").hexdigest()
    if actual_hash != manifest["sha256"] or path.stat().st_size != manifest["size_bytes"]:
        raise ValueError("모델 파일이 기록한 버전과 다릅니다. 출처와 해시를 확인해주세요.")
    with zipfile.ZipFile(path) as archive:
        expected = {"metadata.json", "config.json", "model.weights.h5"}
        if not expected.issubset(archive.namelist()):
            raise ValueError("Keras 저장 파일의 필수 항목이 없습니다.")
        metadata = json.loads(archive.read("metadata.json"))
        config = json.loads(archive.read("config.json"))
    if metadata.get("keras_version") != manifest["saved_keras_version"]:
        raise ValueError("기록한 Keras 저장 버전과 다릅니다.")
    layer = next((item for item in config["config"]["layers"]
                  if item.get("class_name") == manifest["custom_layer"]["class_name"]), None)
    if layer is None:
        raise ValueError("AddPositionEmbedding 레이어를 찾을 수 없습니다.")
    for name in ("max_len", "d_model"):
        if layer["config"].get(name) != manifest["custom_layer"][name]:
            raise ValueError(f"커스텀 레이어의 {name} 설정이 기록과 다릅니다.")
    return {
        "model_file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": actual_hash,
        "saved_keras_version": metadata["keras_version"],
        "custom_layer": {key: layer["config"].get(key) for key in ("name", "max_len", "d_model")},
        "status": "archive_checked_only",
        "weights_loaded": False,
        "prediction_executed": False,
    }


def main():
    try:
        print(json.dumps(inspect(), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        print(f"확인 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
