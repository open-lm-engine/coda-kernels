import re
import json
import matplotlib.pyplot as plt
from collections import defaultdict

RULES = {
    "attention": [
        r"cudnn.*sdpa.*flash",
        r"cudnn::fusion::compute_dot_do_o",
        r"cudnn::fusion::convert_dq_to_16bits",
        r"cudnn::fusion::fmha_reduce",
        r"scaled_dot_product_cudnn_attention",
    ],
    "gemm": [
        r"^nvjet_",
    ],
    "optimizer": [
        r"multi_tensor_apply_kernel",
    ],
}

_COMPILED = [
    (re.compile(pattern), bucket)
    for bucket, patterns in RULES.items()
    for pattern in patterns
]


def classify(name: str) -> str:
    matches = [bucket for pattern, bucket in _COMPILED if pattern.search(name)]
    if len(matches) == 0:
        return "other"
    if len(matches) == 1:
        return matches[0]
    else:
        raise ValueError(f"kernel {name!r} matched multiple rules: {matches}")


def categorize(trace_path: str) -> dict:
    """Walk a Kineto trace and bucket GPU time. Returns:

        {
          "total_us": float,
          "buckets": {
              <bucket>: {
                  "time_us": float,
                  "count": int,
                  "kernels": {<kernel_name>: time_us, ...},
              },
              ...
          },
        }
    """
    with open(trace_path) as f:
        data = json.load(f)

    total_time = 0.0
    bucket_time = defaultdict(float)
    bucket_count = defaultdict(int)
    bucket_kernels = defaultdict(lambda: defaultdict(float))

    for event in data["traceEvents"]:
        category = event.get("cat")
        duration = event.get("dur")

        if category == "kernel":
            name = event.get("name")
            bucket = classify(name)
        elif category in ("gpu_memset", "gpu_memcpy"):
            name = category
            bucket = "other"
        else:
            continue

        total_time = total_time + duration
        bucket_time[bucket] = bucket_time[bucket] + duration
        bucket_count[bucket] = bucket_count[bucket] + 1
        bucket_kernels[bucket][name] = bucket_kernels[bucket][name] + duration

    return {
        "total_time": total_time,
        "buckets": {
            b: {
                "time": bucket_time[b],
                "count": bucket_count[b],
                "kernels": bucket_kernels[b],
            }
            for b in bucket_time.keys()
        },
    }


def visualize(
    trace_path_a: str,
    trace_path_b: str,
    label_a: str,
    label_b: str,
    save_path: str | None = None,
) -> None:
    color_map = {
        "gemm":      "#e1e5f2",
        "attention": "#bfdbf7",
        "optimizer": "#1f7a8c",
        "other":     "#022b3a",
    }

    result_a = categorize(trace_path_a)
    result_b = categorize(trace_path_b)

    buckets = ["gemm", "attention", "optimizer", "other"]
    total_a = result_a["total_time"]
    total_b = result_b["total_time"]
    values_a = [result_a["buckets"][b]["time"] / total_a * 100.0 for b in buckets]
    values_b = [result_b["buckets"][b]["time"] / total_b * 100.0 for b in buckets]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    bar_width = 0.9
    bottom_a = 0.0
    bottom_b = 0.0

    for bucket, value_a, value_b in zip(buckets, values_a, values_b):
        ax.bar(0, value_a, width=bar_width, bottom=bottom_a, color=color_map[bucket], linewidth=0)
        ax.bar(1, value_b, width=bar_width, bottom=bottom_b, color=color_map[bucket], linewidth=0)
        text_color = "white" if bucket == "other" else "black"
        if value_a > 3:
            ax.text(0, bottom_a + value_a / 2, f"{bucket}", ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)
        if value_b > 3:
            ax.text(1, bottom_b + value_b / 2, f"{bucket}", ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)
        bottom_a = bottom_a + value_a
        bottom_b = bottom_b + value_b

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b], fontsize=13)
    ax.set_ylabel("% of total GPU time", fontsize=13)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
