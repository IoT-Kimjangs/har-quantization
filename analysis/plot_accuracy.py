
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 1. 프로젝트 경로
project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"
output_dir = results_dir / "figures"

output_dir.mkdir(parents=True, exist_ok=True)

# 2. 정확도 결과 파일 경로
metrics_files = {
    "WISDM": (
        results_dir
        / "2026-11-10-macos-wisdm-quantization"
        / "metrics.json"
    ),
    "PAMAP2": (
        results_dir
        / "2026-11-10-macos-pamap2-models"
        / "metrics.json"
    )
}

# 3. 모델 이름
model_names = ["FP32", "FP16", "INT8 Dynamic", "INT8 Full"]
variant_keys = ["fp16", "int8_dynamic", "int8_full"]

# 4. 데이터셋별 정확도 분석
for dataset, metrics_path in metrics_files.items():

    # JSON 읽기
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # WISDM과 PAMAP2의 JSON 구조 차이 처리
    if "fp32_baseline" in data:
        fp32_accuracy = data["fp32_baseline"]["accuracy_percent"]
    else:
        fp32_accuracy = data["variants"]["fp32"]["accuracy_percent"]

    # 양자화 모델의 정확도
    accuracies = [fp32_accuracy]

    for key in variant_keys:
        accuracies.append(
            data["variants"][key]["accuracy_percent"]
        )

    # 5. FP32 대비 정확도 변화 계산
    differences = [
        accuracy - fp32_accuracy
        for accuracy in accuracies
    ]

    print(f"\n[{dataset} 정확도 분석]")

    for name, accuracy, diff in zip(
        model_names, accuracies, differences
    ):
        print(
            f"{name}: {accuracy:.4f}% "
            f"({diff:+.4f}%p)"
        )

    # 6. 그래프 생성
    plt.figure(figsize=(9, 5))
    bars = plt.bar(model_names, differences)

    plt.axhline(0, color="black", linewidth=1)

    # 수치 표시
    for bar, diff in zip(bars, differences):
        plt.annotate(
            f"{diff:+.4f}%p",
            (bar.get_x() + bar.get_width() / 2, diff),
            xytext=(0, 6 if diff >= 0 else -15),
            textcoords="offset points",
            ha="center"
        )

    # 데이터셋마다 변화 범위가 달라 자동 설정
    max_diff = max(abs(value) for value in differences)
    limit = max(max_diff * 1.5, 0.01)

    plt.ylim(-limit, limit)

    plt.title(f"{dataset} - Accuracy Change from TFLite FP32")
    plt.xlabel("Quantization Method")
    plt.ylabel("Accuracy Difference (%p)")
    plt.tight_layout()

    # 7. 이미지 저장
    save_path = (
        output_dir / f"{dataset.lower()}_accuracy_change.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")
