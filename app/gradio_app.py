from collections import Counter

import gradio as gr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from core.config import CLASS_COLORS, CLASS_DESCRIPTIONS
from services.model_service import load_model, run_inference, get_cfg, get_encoder
from services.preprocessing import preprocess_signal

load_model()


def make_rhythm_strip(signal: np.ndarray, pred_classes: list, confidences: list):
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=False)

    axes[0].plot(signal, linewidth=0.6, color="steelblue")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw ECG signal")
    axes[0].grid(alpha=0.2)

    time_axis = np.arange(len(pred_classes))
    for t, cls in enumerate(pred_classes):
        axes[1].axvspan(t, t + 1, color=CLASS_COLORS[cls], alpha=0.85)
    axes[1].set_ylabel("Predicted class")
    axes[1].set_yticks([])
    axes[1].set_title("Rhythm strip (predicted class per beat window)")
    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    axes[1].legend(handles=legend_handles, loc="upper right", ncol=5, fontsize=8)

    axes[2].plot(time_axis, confidences, color="gray", linewidth=0.8)
    axes[2].fill_between(time_axis, confidences, alpha=0.3, color="gray")
    axes[2].set_ylabel("Confidence")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Beat window index")
    axes[2].set_title("Model confidence per window")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    return fig


def predict(file):
    if file is None:
        return "Please upload a CSV file.", None, None

    try:
        signal = np.loadtxt(file.name, delimiter=",")
        if signal.ndim > 1:
            signal = signal[:, 0]
        signal = signal.astype(np.float32)
    except Exception as e:
        return f"Error reading file: {e}", None, None

    cfg = get_cfg()

    if len(signal) < cfg["window_size"]:
        return f"Signal too short. Minimum {cfg['window_size']} samples required.", None, None

    filtered, segments = preprocess_signal(signal, cfg["window_size"])

    if len(segments) == 0:
        return "Could not extract any segments from signal.", None, None

    result = run_inference(segments)
    pred_classes = result["predicted_classes"]
    confidences = result["confidences"]

    counts = Counter(pred_classes)
    total = len(pred_classes)
    summary_lines = [f"**Total windows analyzed: {total}**\n"]
    for cls in get_encoder().classes_.tolist():
        if cls in counts:
            pct = counts[cls] / total * 100
            summary_lines.append(
                f"- **{cls}** ({CLASS_DESCRIPTIONS[cls]}): {counts[cls]} windows ({pct:.1f}%)"
            )
    dominant = counts.most_common(1)[0][0]
    summary_lines.append(f"\n**Dominant rhythm: {dominant} — {CLASS_DESCRIPTIONS[dominant]}**")
    summary = "\n".join(summary_lines)

    detail_rows = [
        f"Window {i+1}: **{cls}** ({conf*100:.1f}%)"
        for i, (cls, conf) in enumerate(zip(pred_classes, confidences))
    ]
    detail = "\n".join(detail_rows[:20])
    if total > 20:
        detail += f"\n... and {total - 20} more windows"

    fig = make_rhythm_strip(filtered, pred_classes, confidences)
    return summary, fig, detail


with gr.Blocks(title="ECG Arrhythmia Detection") as demo:
    gr.Markdown("# ECG Arrhythmia Detection")
    gr.Markdown(
        "Upload a single-column CSV file containing raw ECG signal values (360 Hz). "
        "The model classifies each beat window into one of five arrhythmia classes."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload ECG CSV file", file_types=[".csv"])
            predict_btn = gr.Button("Analyse", variant="primary")
            gr.Markdown(
                "**Class legend:**\n"
                "- N: Normal sinus rhythm\n"
                "- L: Left bundle branch block\n"
                "- R: Right bundle branch block\n"
                "- V: Premature ventricular contraction\n"
                "- A: Atrial premature contraction"
            )

        with gr.Column(scale=2):
            summary_out = gr.Markdown(label="Summary")
            plot_out = gr.Plot(label="Rhythm strip")
            detail_out = gr.Markdown(label="Per-window detail")

    predict_btn.click(
        fn=predict,
        inputs=[file_input],
        outputs=[summary_out, plot_out, detail_out],
    )

    gr.Markdown(
        "**Model:** CNN-BiLSTM trained on MIT-BIH Arrhythmia Database · "
        "[HuggingFace Model](https://huggingface.co/dheerajthuvara/ecg-arrhythmia-detection) · "
        "[Portfolio](https://dtlabs.me)"
    )

if __name__ == "__main__":
    demo.launch()
