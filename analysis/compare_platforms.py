
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# 프로젝트 폴더 경로
project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"
output_dir = results_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# 비교할 모델
models = [
    "tflite_fp32",
    "tflite_fp16",
    "tflite_int8_dynamic",
    "tflite_int8_full"
]

labels = ["FP32", "FP16", "INT8 Dynamic", "INT8 Full"]


# 1. summary.csv에서 측정값 가져오기
def load_latency(platform, dataset):
    files = [
        p for p in results_dir.rglob("summary.csv")
        if platform in str(p.parent).lower()
        and dataset.lower() in str(p.parent).lower()
    ]

    if len(files) != 1:
        raise FileNotFoundError(
            f"{platform} / {dataset} 결과 파일 확인 필요: {files}"
        )

    with files[0].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # thread_mode가 default인 결과만 선택
    data = {
        row["variant"]: float(row["median_ms"])
        for row in rows
        if row["thread_mode"] == "default"
    }

    return [data[model] for model in models]


# 2. 데이터셋별 Mac vs Windows 그래프
for dataset in ["WISDM", "PAMAP2"]:

    mac = load_latency("macos", dataset)
    windows = load_latency("windows", dataset)

    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        [i - width / 2 for i in x],
        mac,
        width=width,
        label="Mac"
    )

    plt.bar(
        [i + width / 2 for i in x],
        windows,
        width=width,
        label="Windows"
    )

    plt.xticks(x, labels)
    plt.xlabel("Quantization Method")
    plt.ylabel("Median Inference Latency (ms)")
    plt.title(f"{dataset} - Mac vs Windows")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / f"{dataset.lower()}_mac_vs_windows.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


# 3. INT8 Dynamic의 추론시간 감소율 계산
for dataset in ["WISDM", "PAMAP2"]:
    print(f"\n[{dataset}]")

    for platform in ["macos", "windows"]:
        latencies = load_latency(platform, dataset)

        fp32 = latencies[0]
        int8_dynamic = latencies[2]

        reduction = (fp32 - int8_dynamic) / fp32 * 100

        print(
            f"{platform}: FP32 {fp32:.4f} ms -> "
            f"INT8 Dynamic {int8_dynamic:.4f} ms "
            f"({reduction:.1f}% 감소)"
        )

# 4. 원논문의 TFLite 추론시간 (ms)
# 출처: 원논문 Table 7, Table 8
paper_latency = {
    "WISDM": [0.22, 0.21, 0.86, 9.73],
    "PAMAP2": [0.33, 0.36, 0.26, 2.40]
}

# 5. 원논문 vs Mac vs Windows 비교 그래프
for dataset in ["WISDM", "PAMAP2"]:

    paper = paper_latency[dataset]
    mac = load_latency("macos", dataset)
    windows = load_latency("windows", dataset)

    x = list(range(len(models)))

    plt.figure(figsize=(10, 5))

    plt.plot(x, paper, marker="o", label="Paper")
    plt.plot(x, mac, marker="s", label="Mac")
    plt.plot(x, windows, marker="^", label="Windows")

    plt.xticks(x, labels)

    # 값의 차이가 크므로 로그 스케일 사용
    plt.yscale("log")

    plt.xlabel("Quantization Method")
    plt.ylabel("Inference Latency (ms, log scale)")
    plt.title(f"{dataset} - Paper vs Mac vs Windows")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = output_dir / f"{dataset.lower()}_paper_vs_platforms.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")

# 6. 실제 측정한 모델 파일 크기 읽기
def load_model_sizes(platform, dataset):
    files = [
        p for p in results_dir.rglob("summary.csv")
        if platform in str(p.parent).lower()
        and dataset.lower() in str(p.parent).lower()
    ]

    if len(files) != 1:
        raise FileNotFoundError(f"결과 파일 확인 필요: {files}")

    with files[0].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        row["variant"]: int(row["model_bytes"])
        for row in rows
        if row["thread_mode"] == "default"
    }


# 7. FP32 대비 모델 크기 감소율 계산
for dataset in ["WISDM", "PAMAP2"]:
    sizes = load_model_sizes("macos", dataset)

    fp32_size = sizes["tflite_fp32"]

    print(f"\n[{dataset} 모델 크기 감소율]")

    for model in models[1:]:
        model_size = sizes[model]

        reduction = (fp32_size - model_size) / fp32_size * 100

        print(f"{model}: {reduction:.1f}% 감소")


# 8. 실제 실행 메모리 사용량 분석

def load_memory(platform, dataset):

    # 해당 플랫폼과 데이터셋의 summary.csv 찾기
    files = [
        p for p in results_dir.rglob("summary.csv")
        if platform in p.parent.name.lower()
        and dataset.lower() in p.parent.name.lower()
    ]

    if len(files) != 1:
        raise FileNotFoundError(
            f"{platform} / {dataset} 파일 확인 필요: {files}"
        )

    with files[0].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 플랫폼별 메모리 측정 항목 선택
    if platform == "macos":
        memory_column = "native_peak_rss_bytes"
    else:
        memory_column = "peak_working_set_bytes"

    # TFLite 모델 중 default 실행 결과만 선택
    memory = {
        row["variant"]: int(row[memory_column])
        for row in rows
        if row["thread_mode"] == "default"
        and row["variant"] in models
    }

    # bytes -> MiB 변환
    return {
        model: memory[model] / (1024 * 1024)
        for model in models
    }


# 9. FP32 대비 메모리 변화량 계산 및 그래프 생성

for dataset in ["WISDM", "PAMAP2"]:

    print(f"\n[{dataset} 메모리 사용량 분석]")

    memory_changes = {}

    for platform in ["macos", "windows"]:

        memory = load_memory(platform, dataset)

        fp32_memory = memory["tflite_fp32"]

        changes = [
            memory[model] - fp32_memory
            for model in models
        ]

        memory_changes[platform] = changes

        print(f"\n{platform.upper()} - FP32: {fp32_memory:.2f} MiB")

        for model, change in zip(models, changes):
            print(f"{model}: {change:+.3f} MiB")

    # 그래프 만들기
    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        [i - width / 2 for i in x],
        memory_changes["macos"],
        width=width,
        label="Mac"
    )

    plt.bar(
        [i + width / 2 for i in x],
        memory_changes["windows"],
        width=width,
        label="Windows"
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(x, labels)
    plt.xlabel("Quantization Method")
    plt.ylabel("Memory Change from FP32 (MiB)")
    plt.title(f"{dataset} - Memory Change from TFLite FP32")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / f"{dataset.lower()}_memory_change.png"

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"\nSaved: {save_path}")
