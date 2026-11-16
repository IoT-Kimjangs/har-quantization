
from pathlib import Path
import matplotlib.pyplot as plt

# 1. 양자화 방식
models = ["FP32", "FP16", "INT8 Dynamic", "INT8 Full"]

# 2. 원논문에서 가져온 모델 크기 (KB)
paper_data = {
    "WISDM": [295.20, 163.63, 110.05, 116.52],
    "PAMAP2": [995.71, 520.55, 314.27, 325.07]
}

# 3. 그래프 저장 폴더
output_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# 4. 데이터셋별 그래프 생성
for dataset, sizes in paper_data.items():

    plt.figure(figsize=(9, 5))

    bars = plt.bar(models, sizes)

    # 막대 위에 수치 표시
    for bar, size in zip(bars, sizes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.02,
            f"{size:.2f}",
            ha="center"
        )

    plt.title(f"{dataset} - TFLite Model Size Comparison")
    plt.xlabel("Quantization Method")
    plt.ylabel("Model Size (KB)")
    plt.ylim(0, max(sizes) * 1.2)

    plt.tight_layout()

    # 그래프 이미지 저장
    save_path = output_dir / f"{dataset.lower()}_model_size.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")



# 5. 원논문의 추론시간 데이터 (ms)
paper_latency = {
    "WISDM": [0.22, 0.21, 0.86, 9.73],
    "PAMAP2": [0.33, 0.36, 0.26, 2.40]
}

# 6. 데이터셋별 추론시간 그래프 생성
for dataset, latencies in paper_latency.items():

    plt.figure(figsize=(9, 5))

    bars = plt.bar(models, latencies)

    for bar, latency in zip(bars, latencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.02,
            f"{latency:.2f}",
            ha="center"
        )

    plt.title(f"{dataset} - TFLite Inference Latency")
    plt.xlabel("Quantization Method")
    plt.ylabel("Inference Latency (ms)")
    plt.ylim(0, max(latencies) * 1.2)

    plt.tight_layout()

    save_path = output_dir / f"{dataset.lower()}_latency.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")

