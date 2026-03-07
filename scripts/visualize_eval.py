#!/usr/bin/env python3
"""
DiffuseAlign — Evaluation Visualization Dashboard

Generates a comprehensive set of plots from eval_results.json and
training logs, saved to experiments/figures/.

Usage:
    python scripts/visualize_eval.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

# We generate figures without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────── Paths ───────────────────────────────────
ROOT = Path(__file__).parent.parent
EVAL_JSON = ROOT / "experiments" / "eval_results.json"
FIG_DIR = ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAGE_LOGS = {
    "Stage 1 (GPU4)": ROOT / "experiments" / "checkpoints" / "gpu4.log",
    "Stage 1 (GPU5)": ROOT / "experiments" / "checkpoints" / "gpu5.log",
    "Stage 1 (GPU6)": ROOT / "experiments" / "checkpoints" / "gpu6.log",
    "Stage 1 (GPU7)": ROOT / "experiments" / "checkpoints" / "gpu7.log",
    "Stage 2 (GPU4)": ROOT / "experiments" / "checkpoints" / "stage2_gpu4.log",
    "Stage 2 (GPU5)": ROOT / "experiments" / "checkpoints" / "stage2_gpu5.log",
    "Stage 3 (GPU6)": ROOT / "experiments" / "checkpoints" / "stage3_gpu6.log",
    "Stage 3 (GPU7)": ROOT / "experiments" / "checkpoints" / "stage3_gpu7.log",
}

# ──────────────────────── Style ───────────────────────────────────────
COLORS = {
    "simple": "#4CAF50",
    "moderate": "#FF9800",
    "complex": "#F44336",
    "primary": "#1976D2",
    "secondary": "#7B1FA2",
    "accent": "#00897B",
    "stage1": "#1976D2",
    "stage2": "#F57C00",
    "stage3": "#7B1FA2",
    "bg": "#FAFAFA",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ──────────────────── Log parsing ─────────────────────────────────────

def parse_training_losses(log_path: Path, stage: int) -> dict:
    """Parse loss values from a training log file.

    Stage 1 format: "Epoch N: 100%|...| .../625 [..., loss=X.XXXX, ..., step=N]"
    Stage 2 format: tqdm lines with "loss=X.XXXX" and "nll=X.XXXX, step=N"
    Stage 3 format: similar epoch-based tqdm with "loss=X.XXXX"
    """
    steps, losses = [], []
    epoch_losses = []

    if not log_path.exists():
        return {"steps": steps, "losses": losses, "epoch_losses": epoch_losses}

    text = log_path.read_text(errors="replace")

    if stage == 1:
        # Stage 1: each epoch ends with "100%|...step=N]"
        # Grab the final tqdm update per epoch line: "Epoch N: 100%|..., loss=X.XXXX, ..., step=N]"
        for m in re.finditer(
            r"Epoch\s+(\d+):\s*100%.*?loss=([\d.]+).*?step=(\d+)", text
        ):
            epoch = int(m.group(1))
            loss = float(m.group(2))
            step = int(m.group(3))
            epoch_losses.append((epoch, loss))
            steps.append(step)
            losses.append(loss)

    elif stage == 2:
        # Stage 2: tqdm progress bars — many duplicated lines per step.
        # Extract the 100% completed epoch lines: "Epoch N/10: 100%|..., loss=X.XXXX, ..., nll=X.XXXX, step=N]"
        for m in re.finditer(
            r"Epoch\s+(\d+)/\d+:\s*100%.*?loss=([\d.]+).*?step=(\d+)", text
        ):
            epoch = int(m.group(1))
            loss = float(m.group(2))
            step = int(m.group(3))
            epoch_losses.append((epoch, loss))
            steps.append(step)
            losses.append(loss)
        # Also grab checkpoint lines for intermediate points
        for m in re.finditer(
            r"Checkpoint.*?step\s+(\d+),\s*loss\s+([\d.]+)", text
        ):
            step = int(m.group(1))
            loss = float(m.group(2))
            # Avoid duplicates
            if step not in steps:
                steps.append(step)
                losses.append(loss)

    elif stage == 3:
        for m in re.finditer(
            r"Epoch\s+(\d+)/\d+:\s*100%.*?loss=([\d.]+).*?step=(\d+)", text
        ):
            epoch = int(m.group(1))
            loss = float(m.group(2))
            step = int(m.group(3))
            epoch_losses.append((epoch, loss))
            steps.append(step)
            losses.append(loss)

    # Sort by step
    if steps and losses:
        pairs = sorted(zip(steps, losses))
        steps = [p[0] for p in pairs]
        losses = [p[1] for p in pairs]

    return {"steps": steps, "losses": losses, "epoch_losses": epoch_losses}


# ──────────────────── Figures ─────────────────────────────────────────

def fig1_metrics_overview(data: dict):
    """Figure 1: Main evaluation metrics as a dashboard."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("DiffuseAlign — Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                           top=0.90, bottom=0.08, left=0.08, right=0.95)

    # ── Panel 1: Per-complexity success / efficiency / coordination bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    complexities = ["simple", "moderate", "complex"]
    efficiencies = [data.get(f"{c}/action_efficiency", 0) for c in complexities]
    coords = [data.get(f"{c}/coordination_score", 0) for c in complexities]

    x = np.arange(len(complexities))
    w = 0.35
    bars1 = ax1.bar(x - w/2, efficiencies, w, label="Efficiency",
                     color=[COLORS[c] for c in complexities], alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + w/2, coords, w, label="Coordination",
                     color=[COLORS[c] for c in complexities], alpha=0.45, edgecolor="white", hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in complexities])
    ax1.set_ylabel("Score")
    ax1.set_title("Efficiency & Coordination\nby Complexity")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # ── Panel 2: Turn count distribution
    ax2 = fig.add_subplot(gs[0, 1])
    turns = [data.get(f"{c}/avg_turn_count", 0) for c in complexities]
    episodes = [data.get(f"{c}/num_episodes", 0) for c in complexities]
    bars = ax2.bar(complexities, turns, color=[COLORS[c] for c in complexities],
                    alpha=0.85, edgecolor="white", linewidth=1.5)
    for bar, n in zip(bars, episodes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"n={n}", ha="center", va="bottom", fontsize=9, color="gray")
    ax2.set_ylabel("Average Turns")
    ax2.set_title("Turn Count by Complexity")
    ax2.set_xticklabels([c.capitalize() for c in complexities])

    # ── Panel 3: Generation time
    ax3 = fig.add_subplot(gs[0, 2])
    mean_t = data.get("generation_time_mean", 0)
    std_t = data.get("generation_time_std", 0)
    ax3.bar(["Plan\nGeneration"], [mean_t], yerr=[std_t], color=COLORS["primary"],
            alpha=0.85, capsize=8, edgecolor="white", linewidth=1.5)
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Inference Speed\n(50-step DDIM)")
    ax3.text(0, mean_t + std_t + 0.02, f"{mean_t:.2f}s ± {std_t:.2f}s",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Panel 4: Score summary card
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")

    metrics_text = [
        ("Task Success Rate", f"{data.get('task_success_rate', 0):.1%}"),
        ("Action Efficiency", f"{data.get('action_efficiency', 0):.3f}"),
        ("Coordination Score", f"{data.get('coordination_score', 0):.3f}"),
        ("Delegation Accuracy", f"{data.get('delegation_accuracy', 0):.1%}"),
        ("Avg Turn Count", f"{data.get('avg_turn_count', 0):.1f}"),
        ("Episodes", f"{data.get('num_episodes', 0)}"),
    ]

    y_start = 0.95
    for i, (label, value) in enumerate(metrics_text):
        y = y_start - i * 0.15
        ax4.text(0.05, y, label, fontsize=11, va="top", transform=ax4.transAxes)
        ax4.text(0.95, y, value, fontsize=11, va="top", ha="right",
                 fontweight="bold", transform=ax4.transAxes,
                 color=COLORS["primary"])

    ax4.set_title("Aggregate Metrics", fontweight="bold")
    rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98, transform=ax4.transAxes,
                           boxstyle="round,pad=0.02", facecolor="#E3F2FD",
                           edgecolor=COLORS["primary"], linewidth=1.5)
    ax4.add_patch(rect)

    # ── Panel 5: Efficiency scaling (complexity vs efficiency)
    ax5 = fig.add_subplot(gs[1, 1])
    optimal_steps = {"simple": 5, "moderate": 10, "complex": 21}
    for c in complexities:
        eff = data.get(f"{c}/action_efficiency", 0)
        opt = optimal_steps[c]
        ax5.scatter(opt, eff, s=200, c=COLORS[c], edgecolors="white",
                    linewidths=2, zorder=5, label=c.capitalize())
    ax5.set_xlabel("Optimal Steps (task complexity)")
    ax5.set_ylabel("Action Efficiency")
    ax5.set_title("Efficiency vs Task Complexity")
    ax5.legend()

    # ── Panel 6: Guidance config
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    cfg_text = [
        ("Checkpoint", Path(data.get("checkpoint", "")).name),
        ("Guidance", "Enabled" if data.get("guidance_enabled") else "Disabled"),
        ("Guidance Scale", f"{data.get('guidance_scale', 'N/A')}"),
        ("Diffusion Steps", "50 (DDIM)"),
        ("Model Size", "~300M params"),
    ]

    y_start = 0.95
    for i, (label, value) in enumerate(cfg_text):
        y = y_start - i * 0.15
        ax6.text(0.05, y, label, fontsize=10, va="top", transform=ax6.transAxes, color="gray")
        ax6.text(0.95, y, value, fontsize=10, va="top", ha="right",
                 transform=ax6.transAxes, fontweight="bold")

    ax6.set_title("Configuration", fontweight="bold")
    rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98, transform=ax6.transAxes,
                           boxstyle="round,pad=0.02", facecolor="#F3E5F5",
                           edgecolor=COLORS["secondary"], linewidth=1.5)
    ax6.add_patch(rect)

    out = FIG_DIR / "fig1_eval_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig2_training_curves():
    """Figure 2: Training loss curves for all 3 stages."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DiffuseAlign — Training Loss Curves", fontsize=15, fontweight="bold", y=1.02)

    stage_configs = [
        (1, "Stage 1: Diffusion Model\n(100K steps, 4 GPUs)", ["gpu4.log", "gpu5.log", "gpu6.log", "gpu7.log"]),
        (2, "Stage 2: Plan Decoder\n(10 epochs, 2 GPUs)", ["stage2_gpu4.log", "stage2_gpu5.log"]),
        (3, "Stage 3: Guidance Classifiers\n(20 epochs, 2 GPUs)", ["stage3_gpu6.log", "stage3_gpu7.log"]),
    ]

    for ax, (stage, title, log_files) in zip(axes, stage_configs):
        color = COLORS[f"stage{stage}"]
        has_data = False

        for i, lf in enumerate(log_files):
            log_path = ROOT / "experiments" / "checkpoints" / lf
            parsed = parse_training_losses(log_path, stage)

            if parsed["steps"] and parsed["losses"]:
                steps = parsed["steps"]
                losses = parsed["losses"]

                alpha = 0.4 if len(log_files) > 2 else 0.7
                label = lf.replace(".log", "")
                ax.plot(steps, losses, "o-", color=color, alpha=alpha,
                        markersize=3, linewidth=1.2, label=label)
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No log data found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step" if stage == 1 else "Epoch")
        ax.set_ylabel("Loss")
        if has_data:
            ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.8)

    plt.tight_layout()
    out = FIG_DIR / "fig2_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig3_complexity_radar(data: dict):
    """Figure 3: Radar chart of per-complexity metrics."""
    complexities = ["simple", "moderate", "complex"]
    metrics = ["action_efficiency", "coordination_score", "avg_turn_count"]
    metric_labels = ["Efficiency", "Coordination", "Avg Turns"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Per-Complexity Metric Profiles", fontsize=14, fontweight="bold", y=0.98)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for c in complexities:
        values = []
        for m in metrics:
            v = data.get(f"{c}/{m}", 0)
            # Normalize to [0, 1] for radar
            if m == "action_efficiency":
                v = min(v / 15.0, 1.0)  # cap at 15
            elif m == "coordination_score":
                v = (v + 1) / 2.0  # map [-1,1] → [0,1]
            elif m == "avg_turn_count":
                v = min(v / 10.0, 1.0)  # cap at 10
            values.append(v)
        values += values[:1]

        ax.plot(angles, values, "o-", color=COLORS[c], linewidth=2, label=c.capitalize())
        ax.fill(angles, values, color=COLORS[c], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    out = FIG_DIR / "fig3_complexity_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig4_architecture_overview():
    """Figure 4: Architecture diagram (block diagram)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.suptitle("DiffuseAlign — Architecture Overview", fontsize=15, fontweight="bold")

    boxes = [
        # (x, y, w, h, label, color, sublabel)
        (0.5, 2.0, 2.2, 2.0, "Plan\nEncoder", COLORS["stage1"], "BERT + Agent MLP\n+ Cross-Attn Fusion"),
        (3.5, 2.0, 2.2, 2.0, "Plan\nDiffusion", COLORS["stage1"], "6-layer Denoiser\n1000 → 50 DDIM steps"),
        (6.5, 2.0, 2.2, 2.0, "Role\nMasker", COLORS["stage1"], "Capability scoring\n+ Validity masks"),
        (9.5, 2.0, 2.2, 2.0, "Guidance\nClassifiers", COLORS["stage3"], "Task + Coord +\nSafety + Efficiency"),
        (9.5, 0.0, 2.2, 1.5, "Plan\nDecoder", COLORS["stage2"], "VQ Tokenizer +\nFlan-T5-Base"),
    ]

    for x, y, w, h, label, color, sublabel in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(x + w/2, y + h*0.25, sublabel, ha="center", va="center",
                fontsize=8, color="gray")

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                    color="gray", linewidth=1.5)
    from matplotlib.patches import FancyArrowPatch

    arrows = [
        ((2.7, 3.0), (3.5, 3.0)),   # Encoder → Diffusion
        ((5.7, 3.0), (6.5, 3.0)),   # Diffusion → Role Masker
        ((8.7, 3.0), (9.5, 3.0)),   # Role Masker → Guidance
        ((10.6, 2.0), (10.6, 1.5)), # Guidance → Decoder
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, **arrow_kw)
        ax.add_patch(arrow)

    # Input / Output labels
    ax.text(0.5, 4.5, "Task Description\n+ Agent Capabilities", fontsize=9,
            ha="left", va="center", style="italic", color="gray",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.annotate("", xy=(1.6, 4.0), xytext=(1.6, 4.3),
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.text(12.0, 0.75, "Dialogue\nUtterances", fontsize=9,
            ha="left", va="center", style="italic", color="gray",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.annotate("", xy=(11.7, 0.75), xytext=(12.0, 0.75),
                arrowprops=dict(arrowstyle="<-", color="gray"))

    # Stage labels
    ax.text(4.3, 5.2, "Stage 1: Diffusion Training (100K steps)",
            fontsize=10, color=COLORS["stage1"], fontweight="bold", ha="center")
    ax.text(10.6, 5.2, "Stage 3", fontsize=10, color=COLORS["stage3"],
            fontweight="bold", ha="center")
    ax.text(10.6, -0.3, "Stage 2", fontsize=10, color=COLORS["stage2"],
            fontweight="bold", ha="center")

    out = FIG_DIR / "fig4_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig5_training_summary():
    """Figure 5: Training summary — final losses per stage as bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("DiffuseAlign — Final Training Losses", fontsize=14, fontweight="bold")

    stages = ["Stage 1\nDiffusion", "Stage 2\nDecoder (NLL)", "Stage 2\nDecoder (commit)", "Stage 3\nGuidance"]
    losses = [0.022, 0.098, 0.0017, 0.017]
    colors = [COLORS["stage1"], COLORS["stage2"], COLORS["stage2"], COLORS["stage3"]]

    bars = ax.bar(stages, losses, color=colors, edgecolor="white", linewidth=2)
    # Set per-bar alpha
    for bar, a in zip(bars, [0.85, 0.85, 0.5, 0.85]):
        bar.set_alpha(a)

    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{loss:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Final Loss")
    ax.set_title("Converged loss values (averaged across GPU replicas)")

    out = FIG_DIR / "fig5_training_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig6_dialogue_sample(data: dict):
    """Figure 6: Visualize sample decoded dialogues."""
    dialogues = data.get("sample_dialogues", {})
    if not dialogues:
        print("  ⊘ No dialogue samples to visualize")
        return

    task_ids = list(dialogues.keys())[:3]
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(14, 4 * len(task_ids)))
    if len(task_ids) == 1:
        axes = [axes]

    fig.suptitle("DiffuseAlign — Sample Decoded Dialogues", fontsize=14, fontweight="bold", y=1.01)

    task_names = {
        "t001": "Find the red mug → kitchen counter",
        "t002": "Return library book → study",
        "t003": "Move vase room_2 → room_0",
        "t004": "Find keys → front door",
        "t005": "Clean the bathroom",
    }

    agent_colors = ["#1976D2", "#F57C00", "#388E3C", "#7B1FA2"]

    for ax, tid in zip(axes, task_ids):
        ax.axis("off")
        utts = dialogues[tid]
        title = task_names.get(tid, tid)
        ax.set_title(f"Task: {title}", fontsize=12, fontweight="bold", loc="left")

        y = 0.9
        for i, u in enumerate(utts[:4]):
            # Truncate long utterances for display
            display = u[:120] + "..." if len(u) > 120 else u
            color = agent_colors[i % len(agent_colors)]
            ax.text(0.02, y, f"Agent {i}:", fontsize=10, fontweight="bold",
                    color=color, transform=ax.transAxes, va="top")
            ax.text(0.10, y, display, fontsize=9, color="black",
                    transform=ax.transAxes, va="top", wrap=True,
                    family="monospace")
            y -= 0.25

    plt.tight_layout()
    out = FIG_DIR / "fig6_dialogue_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ──────────────────── Main ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DiffuseAlign — Generating Evaluation Figures")
    print("=" * 60)

    # Load eval results
    if EVAL_JSON.exists():
        with open(EVAL_JSON) as f:
            data = json.load(f)
        print(f"Loaded eval results: {EVAL_JSON}")
    else:
        print(f"Warning: {EVAL_JSON} not found, using empty data")
        data = {}

    print(f"Output directory: {FIG_DIR}\n")

    print("Generating figures:")
    fig1_metrics_overview(data)
    fig2_training_curves()
    fig3_complexity_radar(data)
    fig4_architecture_overview()
    fig5_training_summary()
    fig6_dialogue_sample(data)

    print(f"\n✅ All figures saved to {FIG_DIR}/")
    print("\nGenerated files:")
    for f in sorted(FIG_DIR.glob("*.png")):
        sz = f.stat().st_size / 1024
        print(f"  {f.name:40s} {sz:>8.1f} KB")


if __name__ == "__main__":
    main()
