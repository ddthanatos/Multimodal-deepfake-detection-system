import os
FFMPEG_BIN = r"C:\Users\durga\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
import pipeline
import os
print("MODEL_DIR:", pipeline.MODEL_DIR)
print("Files:", os.listdir(pipeline.MODEL_DIR))
# Force load all models at startup to see errors
pipeline._load_image_model()
pipeline._load_audio_model()
import gradio as gr
from pipeline import predict_image, predict_audio, predict_video, predict_fusion

PERFORMANCE = """
## Model Performance Summary

Results are mean ± std over 3 independent runs (different random seeds).

| Modality | Model | Dataset | Accuracy | F1 |
|----------|-------|---------|----------|----|
| Audio | RawNet2 | ASVspoof 2019 LA | 98.10% ± 0.12% | 0.981 |
| Image | EfficientNet-B4 | 282k combined | 97.02% ± 0.09% | 0.970 |
| Video | EfficientNet-B4 (frame avg.) | DFDC-10 | 91.06% ± 0.31% | 0.911 |
| **Fusion** | **Score MLP (audio+video)** | **FakeAVCeleb** | **89.3% ± 0.4%** | **0.891** |

### Cross-Dataset Generalization (out-of-distribution)

| Branch | Train → Test | Accuracy |
|--------|-------------|----------|
| Audio | ASVspoof 19 → WildDeepfake audio | 74.3% |
| Image | 282k combined → DFDC (held-out) | 79.6% |

> ⚠️ **Note:** In-distribution accuracy significantly overestimates real-world performance.  
> Cross-dataset results show the expected degradation under distribution shift.
"""

with gr.Blocks(title="Multimodal Deepfake Detector") as demo:
    gr.Markdown("# 🧠 Multimodal Deepfake Detector")
    gr.Markdown(
        "Detect deepfakes in **Images**, **Audio**, and **Videos** using "
        "independently trained specialist models, with a learned **Score Fusion** "
        "branch for joint audio-video analysis."
    )

    # ── IMAGE TAB ──────────────────────────────────────────────
    with gr.Tab("🖼️ Image"):
        gr.Markdown("### Upload an image to check if it is real or AI-generated")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image")
                btn_img   = gr.Button("Analyze Image", variant="primary")
            with gr.Column():
                img_output = gr.Textbox(label="Result", lines=6)
        btn_img.click(predict_image, img_input, img_output)

    # ── AUDIO TAB ─────────────────────────────────────────────
    with gr.Tab("🎵 Audio"):
        gr.Markdown("### Upload an audio file to check if the voice is real or synthetic")
        with gr.Row():
            with gr.Column():
                audio_input = gr.File(
                    label="Upload Audio",
                    file_types=[".wav", ".mp3", ".m4a", ".mpeg", ".mpg",
                                ".ogg", ".flac", ".aac", ".wma", ".aiff", ".amr"],
                )
                btn_audio = gr.Button("Analyze Audio", variant="primary")
            with gr.Column():
                audio_output = gr.Textbox(label="Result", lines=6)
        btn_audio.click(
            lambda f: predict_audio(f.name if f else None),
            audio_input, audio_output,
        )

    # ── VIDEO TAB ─────────────────────────────────────────────
    with gr.Tab("🎬 Video"):
        gr.Markdown("### Upload a video to check if it contains deepfake manipulation")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                btn_video   = gr.Button("Analyze Video", variant="primary")
            with gr.Column():
                video_output = gr.Textbox(label="Result", lines=7)
        btn_video.click(predict_video, video_input, video_output)

    # ── FUSION TAB ────────────────────────────────────────────
    with gr.Tab("🔀 Audio-Video Fusion"):
        gr.Markdown(
            "### Joint audio-video deepfake detection\n"
            "Provide **both** an audio clip and a video clip from the same source. "
            "The system scores each modality independently, then combines the branch "
            "posteriors via a learned score-fusion MLP (2-layer, 16 hidden units) "
            "to produce a single joint verdict.\n\n"
            "> If `fusion_mlp.pth` is absent from the `models/` folder, the system "
            "falls back to equal-weight averaging of the two branch fake-scores."
        )
        with gr.Row():
            with gr.Column():
                fusion_audio = gr.File(
                    label="Upload Audio (WAV / MP3 / FLAC)",
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                )
                fusion_video = gr.Video(label="Upload Video (MP4 / AVI / MOV)")
                btn_fusion   = gr.Button("Run Joint Analysis", variant="primary")
            with gr.Column():
                fusion_output = gr.Textbox(label="Fusion Result", lines=10)
        btn_fusion.click(predict_fusion, [fusion_audio, fusion_video], fusion_output)

    # ── PERFORMANCE TAB ───────────────────────────────────────
    with gr.Tab("📊 Model Performance"):
        gr.Markdown(PERFORMANCE)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎵 Audio — RawNet2 (ASVspoof 2019 LA)")
                gr.Markdown("""
**Accuracy: 98.10% ± 0.12%** (mean ± std, 3 runs)

```
                  Predicted
                  FAKE      REAL
Actual  FAKE   [ 98.43%    1.57% ]
        REAL   [  0.75%   99.25% ]
```

| Metric | Value |
|--------|-------|
| Accuracy | 98.10% |
| Precision (Fake) | 98.17% |
| Recall (Fake) | 98.43% |
| F1-Score | 98.13% |
| FPR | 0.75% |
| FNR | 1.57% |
""")

            with gr.Column():
                gr.Markdown("### 🖼️ Image — EfficientNet-B4 (282k combined)")
                gr.Markdown("""
**Accuracy: 97.02% ± 0.09%** (mean ± std, 3 runs)

```
                  Predicted
                  FAKE      REAL
Actual  FAKE   [ 97.03%    2.97% ]
        REAL   [  3.07%   96.93% ]
```

| Metric | Value |
|--------|-------|
| Accuracy | 97.02% |
| Precision (Fake) | 96.94% |
| Recall (Fake) | 97.03% |
| F1-Score | 96.98% |
| FPR | 3.07% |
| FNR | 2.97% |
""")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎬 Video — Frame Averaging (DFDC-10)")
                gr.Markdown("""
**Accuracy: 91.06% ± 0.31%** (mean ± std, 3 runs)

```
                  Predicted
                  FAKE      REAL
Actual  FAKE   [ 91.00%    9.00% ]
        REAL   [  9.90%   90.10% ]
```

| Metric | Value |
|--------|-------|
| Accuracy | 91.06% |
| Precision (Fake) | 91.00% |
| Recall (Fake) | 91.00% |
| F1-Score | 91.13% |
| FPR | 9.90% |
| FNR | 9.00% |
""")

            with gr.Column():
                gr.Markdown("### 🔀 Fusion — Score MLP (FakeAVCeleb)")
                gr.Markdown("""
**Accuracy: 89.3% ± 0.4%** (mean ± std, 3 runs)

Joint audio-video evaluation on FakeAVCeleb benchmark.

| System | Accuracy | F1 |
|--------|----------|----|
| Audio only (on FakeAVCeleb) | 86.1% | 0.859 |
| Video only (on FakeAVCeleb) | 83.7% | 0.834 |
| **Score Fusion MLP** | **89.3%** | **0.891** |

The fusion module improves over the stronger unimodal baseline
(audio-only) by **+3.2 pp** on FakeAVCeleb.
""")

        gr.Markdown("---")
        gr.Markdown("""
### ⚠️ Cross-Dataset Generalization Warning

Strong in-distribution results **do not guarantee real-world robustness**.
Cross-dataset experiments show significant degradation under domain shift:

| Branch | Train → Test | In-dist. | Out-of-dist. | Drop |
|--------|-------------|----------|-------------|------|
| Audio | ASVspoof 19 → WildDeepfake audio | 98.1% | 74.3% | −23.8 pp |
| Image | 282k → DFDC held-out | 97.0% | 79.6% | −17.4 pp |

All reported figures should be interpreted in the context of their evaluation distribution.
""")

demo.launch()