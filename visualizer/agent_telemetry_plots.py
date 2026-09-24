"""
Visualization module for Multi-Agent Healthcare Telemetry and Triage Metrics.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any


def plot_agent_telemetry_summary(
    telemetry_logs: List[Dict[str, Any]],
    output_dir: str = "outputs/figures"
) -> str:
    """
    Plots multi-agent interaction latency and routing breakdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(telemetry_logs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    if not df.empty and "agent" in df.columns:
        agent_counts = df["agent"].value_counts()
        colors = ["#2b5c8f", "#d95f02", "#7570b3"]
        axes[0].bar(agent_counts.index, agent_counts.values, color=colors[:len(agent_counts)], alpha=0.85, edgecolor="black")
        axes[0].set_title("Multi-Agent Inbound Routing Volume", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Processed Messages", fontsize=11)
        axes[0].set_xticks(range(len(agent_counts)))
        axes[0].set_xticklabels([a.replace("Agent", "") for a in agent_counts.index], rotation=15)
        axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
        for i, v in enumerate(agent_counts.values):
            axes[0].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    if not df.empty and "latency_ms" in df.columns:
        agent_latency = df.groupby("agent")["latency_ms"].mean()
        axes[1].barh(agent_latency.index, agent_latency.values, color="#1b9e77", alpha=0.85, edgecolor="black")
        axes[1].set_title("Mean Agent Routing Latency (ms)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Latency (milliseconds)", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.4, axis="x")
        for i, v in enumerate(agent_latency.values):
            axes[1].text(v + 0.05, i, f"{v:.2f} ms", va="center", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "agent_telemetry_dashboard.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path
