"""
Generate publication-quality figures from evaluation results.

Usage:
    python make_figures.py --results_dirs output_eval_v6 output_eval_novel_v6 ...
"""

import argparse
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def make_comparison_grid(image_dir, output_path, max_images=8, ncols=4):
    """Create a grid of GT vs Rendered comparison images."""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    if not images:
        print(f"  No images in {image_dir}")
        return

    # Sample evenly
    if len(images) > max_images:
        indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
        images = [images[i] for i in indices]

    loaded = [Image.open(os.path.join(image_dir, f)) for f in images]
    w, h = loaded[0].size

    nrows = (len(loaded) + ncols - 1) // ncols
    grid = Image.new("RGB", (w * ncols, h * nrows), (255, 255, 255))

    for idx, img in enumerate(loaded):
        row, col = divmod(idx, ncols)
        grid.paste(img, (col * w, row * h))

    grid.save(output_path, quality=95)
    print(f"  Saved comparison grid: {output_path} ({len(loaded)} images, {nrows}x{ncols})")


def make_psnr_plot(metrics_list, labels, output_path):
    """Create per-frame PSNR plot using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping PSNR plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for metrics, label in zip(metrics_list, labels):
        if "psnr_per_frame" in metrics:
            psnr = metrics["psnr_per_frame"]
        elif "novel_view" in metrics:
            psnr = metrics["novel_view"]["psnr_per_frame"]
        else:
            continue
        ax.plot(range(len(psnr)), psnr, "-o", markersize=3, label=label)

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-Frame Rendering Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PSNR plot: {output_path}")


def make_loss_curve(log_path, output_path):
    """Create training loss curve from train_log.json."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping loss curve")
        return

    with open(log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    losses = [e["loss"] for e in log]
    l1s = [e["l1"] for e in log]
    ssims = [e["ssim"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, losses, label="Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, l1s, label="L1")
    ax2.plot(epochs, ssims, label="SSIM Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.set_title("Loss Components")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss curve: {output_path}")


def make_latex_table(results_dict, output_path):
    """Generate LaTeX table from results dictionary.

    results_dict: {scene_name: {method_name: metrics_dict}}
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Quantitative comparison on streaming Gaussian map construction.}")
    lines.append(r"  \label{tab:main_results}")
    lines.append(r"  \begin{tabular}{l|cccc|cccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{4}{c|}{Same-View} & \multicolumn{4}{c}{Novel-View} \\")
    lines.append(r"    Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & \#GS"
                 r" & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & FPS \\")
    lines.append(r"    \midrule")

    for scene, methods in results_dict.items():
        lines.append(f"    \\multicolumn{{9}}{{l}}{{\\textit{{{scene}}}}} \\\\")
        for method, m in methods.items():
            sv = m.get("same_view", m)
            nv = m.get("novel_view", {})
            sv_psnr = f"{sv.get('psnr', sv.get('avg_psnr', 0)):.2f}"
            sv_ssim = f"{sv.get('ssim', sv.get('avg_ssim', 0)):.4f}"
            sv_lpips = f"{sv.get('lpips', sv.get('avg_lpips', 0)):.4f}" if sv.get('lpips', sv.get('avg_lpips')) else "---"
            n_gs = f"{m.get('total_gaussians', m.get('train_map_gaussians', 0)):,}"
            nv_psnr = f"{nv.get('psnr', 0):.2f}" if nv else "---"
            nv_ssim = f"{nv.get('ssim', 0):.4f}" if nv else "---"
            nv_lpips = f"{nv.get('lpips', 0):.4f}" if nv.get('lpips') else "---"
            fps = f"{m.get('fps', 0):.1f}"
            lines.append(f"    {method} & {sv_psnr} & {sv_ssim} & {sv_lpips} & {n_gs}"
                         f" & {nv_psnr} & {nv_ssim} & {nv_lpips} & {fps} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"  Saved LaTeX table: {output_path}")


def make_nearest_k_bar_chart(output_path):
    """Create bar chart comparing nearest-K rendering modes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping bar chart")
        return

    modes = ["Full Map\n(K=all)", "Nearest-3", "Nearest-1"]
    train_psnr = [22.53, 24.83, 36.70]
    novel_psnr = [22.53, 24.79, 35.21]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bars1 = ax.bar(x - width/2, train_psnr, width, label="Train View", color="#4C72B0")
    bars2 = ax.bar(x + width/2, novel_psnr, width, label="Novel View", color="#DD8452")

    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Effect of Nearest-K Frame Selection (Courthouse)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.set_ylim(18, 40)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nearest-K bar chart: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="figures")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base = "D:/Projects/SLAM-GS/lingbotMap/lingbot-map"

    # Generate comparison grids for all available eval dirs
    eval_dirs = {
        "v8_novel_nearest1": "output_eval_novel_v8_nearest1",
        "v6_novel_nearest1": "output_eval_novel_v6_nearest1",
        "v8_novel_fullmap": "output_eval_novel_v8_50f",
    }
    for name, edir in eval_dirs.items():
        comp_dir = os.path.join(base, edir, "comparisons")
        if os.path.isdir(comp_dir):
            make_comparison_grid(comp_dir, os.path.join(args.output_dir, f"grid_{name}.png"))

    # Generate loss curves
    train_logs = {
        "v6_20f": "output_train_v6/train_log.json",
        "v6_50f_full": "output_train_v6_50f_full/train_log.json",
        "v8_reproject": "output_train_v8_reproject/train_log.json",
    }
    for name, lpath in train_logs.items():
        full_path = os.path.join(base, lpath)
        if os.path.isfile(full_path):
            make_loss_curve(full_path, os.path.join(args.output_dir, f"loss_{name}.png"))

    # Generate per-frame PSNR plots for novel-view evals
    novel_metrics = []
    novel_labels = []
    for name, edir in [
        ("GCA-Splat (v6, 300ep)", "output_eval_novel_v6_full_nearest1"),
        ("+ cross-view (v8)", "output_eval_novel_v8_nearest1"),
    ]:
        mpath = os.path.join(base, edir, "metrics.json")
        if os.path.isfile(mpath):
            novel_metrics.append(load_metrics(mpath))
            novel_labels.append(name)
    if novel_metrics:
        make_psnr_plot(novel_metrics, novel_labels,
                       os.path.join(args.output_dir, "psnr_novel_view.png"))

    # Nearest-K bar chart
    make_nearest_k_bar_chart(os.path.join(args.output_dir, "nearest_k_comparison.png"))

    # LaTeX table with all available results
    results = {}
    for name, edir in [
        ("v8 (cross-view)", "output_eval_novel_v8_nearest1"),
        ("v6 (baseline)", "output_eval_novel_v6_nearest1"),
    ]:
        mpath = os.path.join(base, edir, "metrics.json")
        if os.path.isfile(mpath):
            m = load_metrics(mpath)
            results.setdefault("Courthouse", {})[name] = {
                "same_view": m.get("train_view", {}),
                "novel_view": m.get("novel_view", {}),
                "total_gaussians": m.get("train_map_gaussians", 0),
                "fps": m.get("fps", 0),
            }
    if results:
        make_latex_table(results, os.path.join(args.output_dir, "table_main.tex"))

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
