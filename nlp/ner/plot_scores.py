import json
import os
from collections import Counter

import matplotlib.pyplot as plt


def plot_confidence_distribution(json_file="parsed_dataset.json"):
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    print("Loading data...")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all confidence scores and round to 1 decimal place to group them
    scores = [round(item.get("confidence", 0.0), 1) for item in data]
    score_counts = Counter(scores)

    # Prepare data for plotting
    labels = sorted(score_counts.keys())
    values = [score_counts[label] for label in labels]

    # Print text summary
    print("\n--- 置信度分数分布 ---")
    for label, count in zip(labels, values):
        print(f"分数 {label:.1f}: {count:5d} 条")

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [f"{score_label:.1f}" for score_label in labels],
        values,
        color="#4A90E2",
        edgecolor="black",
    )

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(values) * 0.01),
            int(yval),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title(
        "Media Parser: Confidence Score Distribution", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Confidence Score (0.0 to 1.0)", fontsize=12)
    plt.ylabel("Number of Parsed Items", fontsize=12)

    # Hide top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Save chart
    output_img = "confidence_distribution.png"
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"\n✅ 柱状图已成功绘制并保存至: {output_img}")


if __name__ == "__main__":
    try:
        plot_confidence_distribution()
    except ImportError:
        print("\n缺少 matplotlib 库，请先运行: pip install matplotlib")
