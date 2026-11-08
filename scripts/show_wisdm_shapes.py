"""Step 2: load the verified model and print input/output shapes, without prediction."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Check the archive first; never deserialize a different file silently.
    from inspect_wisdm_model import inspect
    inspect()
    if sys.prefix == sys.base_prefix:
        raise SystemExit("가상환경의 Python으로 실행해주세요: .venv/bin/python scripts/show_wisdm_shapes.py")
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("KERAS_HOME", str(Path(sys.prefix) / ".keras"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(ROOT / "src"))
    import keras
    from har_quantization.layers import AddPositionEmbedding
    model = keras.models.load_model(
        ROOT / "models/wisdm_transformer_best_fp32.keras",
        custom_objects={"AddPositionEmbedding": AddPositionEmbedding},
        compile=False,
        safe_mode=True,
    )
    print("모델 로드 완료")
    print("입력 형태:", model.input_shape)
    print("출력 형태:", model.output_shape)
    print("None은 한 번에 넣는 샘플 수가 고정되지 않았다는 뜻입니다.")
    print("입력은 (샘플 수, 시간 시점 수, 센서 채널 수), 출력은 (샘플 수, 행동 클래스 수)로 읽습니다.")
    print("예측은 아직 실행하지 않았습니다.")


if __name__ == "__main__":
    main()
