"""
Streamlit Web UI Demo for Transformer Summarization + Neural TTS Pipeline.
Features:
- Live interactive text summarizer with BART / Pegasus
- Real-time neural voice synthesizer with 22050Hz audio playback
- Multi-domain conversation explorer (Customer Support, Tech, Healthcare, Finance)
- ROUGE & Semantic Consistency evaluation dashboard
- Interactive Mean Opinion Score (MOS) Human Evaluation Interface
"""

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import load_config
from src.dataset_generator import TEMPLATES
from src.evaluation import Evaluator
from src.preprocessing import preprocess_text
from src.summarization import TextSummarizer
from src.tts_pipeline import TTSPipeline
from src.utils import get_audio_metadata, load_json

# Set Page Config with Rich Title & Icon
st.set_page_config(
    page_title="Neural TTS & Transformer Summarization Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic Dark UI Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .highlight-box {
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline_components():
    """Cache models in memory for instant interactive inference."""
    cfg = load_config("config.yaml")
    summarizer = TextSummarizer(
        model_name=cfg.summarization.model_name,
        device=cfg.summarization.device,
        use_lazy_loading=True,
    )
    tts = TTSPipeline(
        model_name=cfg.tts.model_name,
        sample_rate=cfg.tts.sample_rate,
        device=cfg.tts.device,
    )
    evaluator = Evaluator(
        rouge_metrics=cfg.evaluation.rouge_metrics,
        consistency_threshold=cfg.evaluation.consistency_threshold,
    )
    return cfg, summarizer, tts, evaluator


cfg, summarizer, tts, evaluator = get_pipeline_components()

# Header
st.markdown('<div class="main-title">🎙️ Expressive AI: Transformer Summarization + Neural TTS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production-grade End-to-End pipeline for long-form document compression, speech synthesis, and objective quality scoring.</div>',
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("⚡ Control Panel")
mode = st.sidebar.radio(
    "Select Workspace:",
    [
        "🚀 Live End-to-End Studio",
        "📊 Batch Pipeline & Results",
        "📝 Human MOS Evaluation",
        "⚙️ Pipeline Architecture & Specs",
    ],
)

# -------------------------------------------------------------
# MODE 1: Live Studio
# -------------------------------------------------------------
if mode == "🚀 Live End-to-End Studio":
    st.subheader("Interactive Long-Form Text to Speech")

    col_preset, col_ratio = st.columns([2, 1])
    with col_preset:
        domain_choice = st.selectbox(
            "Load Sample Preset from Domain:",
            ["Custom Input"] + list(TEMPLATES.keys()),
        )

    with col_ratio:
        max_summary_words = st.slider("Target Max Summary Length (words)", 20, 150, 70)

    # Populate text area
    default_text = ""
    if domain_choice != "Custom Input" and domain_choice in TEMPLATES:
        default_text = TEMPLATES[domain_choice][0]
    elif domain_choice == "Custom Input":
        default_text = (
            "Customer Support Interaction: Cloud Infrastructure Billing Discrepancy.\n"
            "Customer reported an uncalculated overage charge of $480 on Kubernetes cluster pods for August. "
            "Telemetry logs indicated that while primary workers were scaled from 16 to 4 nodes, an automated nightly maintenance script triggered spot instances. "
            "The support engineer applied a courtesy credit of $480 to the client account and reconfigured the pod autoscaler."
        )

    user_text = st.text_area("Input Raw Long-Form Document / Transcript:", value=default_text, height=180)

    if st.button("✨ Run End-to-End Pipeline (Summarize + Synthesize + Evaluate)", type="primary"):
        if not user_text.strip():
            st.error("Please provide valid input text.")
        else:
            with st.spinner("1/3 Normalizing & Cleaning Text..."):
                cleaned_text = preprocess_text(user_text)

            with st.spinner("2/3 Generating Transformer Abstractive Summary..."):
                summary = summarizer.summarize(cleaned_text, target_max_length=max_summary_words)

            with st.spinner("3/3 Synthesizing Neural Speech (22050 Hz)..."):
                out_audio_dir = Path("data/audio/generated_speeches")
                out_audio_dir.mkdir(parents=True, exist_ok=True)
                audio_file = str(out_audio_dir / "studio_demo_output.wav")
                meta = tts.synthesize(summary, audio_file)

            # Evaluation
            eval_metrics = evaluator.evaluate_sample(
                sample_id="live_demo",
                original_text=cleaned_text,
                summary_text=summary,
                audio_path=audio_file,
            )

            st.success("✅ End-to-End Processing Complete!")

            # Display Results in 2 Columns
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("### 📄 Generated Abstractive Summary")
                st.info(summary)
                st.caption(
                    f"Original: **{len(cleaned_text.split())} words** | Summary: **{len(summary.split())} words** | Compression: **{eval_metrics['compression_ratio']*100:.1f}%**"
                )

            with c2:
                st.markdown("### 🔊 Synthesized Neural Speech (22050 Hz)")
                if os.path.exists(audio_file):
                    st.audio(audio_file, format="audio/wav")
                    st.caption(
                        f"Duration: **{meta.get('duration_seconds', 0.0)}s** | Engine: **{meta.get('engine_used')}** | Sample Rate: **{meta.get('sample_rate')} Hz**"
                    )

            # Metrics Row
            st.markdown("### 📊 Objective Quality Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{eval_metrics["rouge1_f"]:.2f}</div><div class="metric-lbl">ROUGE-1</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{eval_metrics["rouge2_f"]:.2f}</div><div class="metric-lbl">ROUGE-2</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{eval_metrics["rougeL_f"]:.2f}</div><div class="metric-lbl">ROUGE-L</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{eval_metrics["consistency_score"]*100:.1f}%</div><div class="metric-lbl">Consistency</div></div>',
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{eval_metrics["estimated_mos"]:.2f}/5</div><div class="metric-lbl">Estimated MOS</div></div>',
                    unsafe_allow_html=True,
                )


# -------------------------------------------------------------
# MODE 2: Batch Pipeline & Results
# -------------------------------------------------------------
elif mode == "📊 Batch Pipeline & Results":
    st.subheader("Batch Dataset Processing & Evaluation Dashboard")

    metrics_file = Path("results/evaluation_metrics.json")
    results_json = Path("data/processed/summarized_outputs.json")

    if metrics_file.exists():
        metrics = load_json(str(metrics_file))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Samples", metrics.get("total_samples", 0))
        m2.metric("Avg ROUGE-1", f"{metrics.get('avg_rouge1', 0):.4f}")
        m3.metric("Avg ROUGE-2", f"{metrics.get('avg_rouge2', 0):.4f}")
        m4.metric("Compression", f"{metrics.get('avg_compression_ratio', 0)*100:.1f}%")
        m5.metric("Audio Quality (MOS)", f"{metrics.get('audio_quality_score', 0):.2f} / 5.0")

    if results_json.exists():
        data = load_json(str(results_json))
        df = pd.DataFrame(data)
        st.markdown("### Processed Samples Explorer")
        st.dataframe(
            df[["id", "domain", "original_word_count", "summary_word_count", "compression_ratio", "rouge1_f", "consistency_score", "estimated_mos"]],
            use_container_width=True,
        )

        selected_id = st.selectbox("Select Sample to Inspect & Play:", df["id"].tolist())
        selected_sample = df[df["id"] == selected_id].iloc[0]

        col_orig, col_summ = st.columns(2)
        with col_orig:
            st.markdown("**Original Text:**")
            st.write(selected_sample["original_text"])
        with col_summ:
            st.markdown("**Generated Summary:**")
            st.write(selected_sample["summary"])
            if os.path.exists(selected_sample.get("audio_path", "")):
                st.audio(selected_sample["audio_path"], format="audio/wav")
    else:
        st.info("No batch run detected yet. Run `python main.py --mode pipeline` from terminal or click below to launch.")
        if st.button("Run Pipeline on 25 Samples"):
            from main import run_pipeline
            run_pipeline("data/raw/sample_conversations.json", max_samples=25)
            st.rerun()


# -------------------------------------------------------------
# MODE 3: Human MOS Evaluation
# -------------------------------------------------------------
elif mode == "📝 Human MOS Evaluation":
    st.subheader("Standardized MOS (Mean Opinion Score) Evaluation Protocol")
    st.markdown(
        """
        Subjective quality assessment protocol adhering to ITU-T P.800 recommendations for synthetic speech intelligibility and semantic naturalness.
        """
    )

    audio_files = list(Path("data/audio/generated_speeches").glob("*.wav"))
    if not audio_files:
        st.warning("No generated audio files found in `data/audio/generated_speeches/`. Run the pipeline first.")
    else:
        selected_audio = st.selectbox("Select Audio Sample for Human Rating:", [str(p) for p in audio_files])
        st.audio(selected_audio, format="audio/wav")

        with st.form("mos_form"):
            st.markdown("#### Subjective Quality Criteria (Scale: 1 = Bad, 5 = Excellent)")
            c1, c2, c3 = st.columns(3)
            with c1:
                q1 = st.slider("1. Speech Intelligibility & Clarity", 1, 5, 4)
            with c2:
                q2 = st.slider("2. Naturalness & Prosody", 1, 5, 4)
            with c3:
                q3 = st.slider("3. Semantic Consistency & Alignment", 1, 5, 5)

            notes = st.text_input("Evaluator Notes (Optional):")
            submitted = st.form_submit_button("Submit MOS Rating")

            if submitted:
                overall = round((q1 + q2 + q3) / 3.0, 2)
                mos_csv = Path("results/mos_scores.csv")
                row = {
                    "sample_id": Path(selected_audio).stem,
                    "intelligibility": q1,
                    "naturalness": q2,
                    "consistency": q3,
                    "overall_score": overall,
                    "notes": notes,
                }
                if mos_csv.exists():
                    df_mos = pd.read_csv(mos_csv)
                    df_mos = pd.concat([df_mos, pd.DataFrame([row])], ignore_index=True)
                else:
                    df_mos = pd.DataFrame([row])
                df_mos.to_csv(mos_csv, index=False)
                st.success(f"Rating submitted! Overall sample score: **{overall} / 5.0**")


# -------------------------------------------------------------
# MODE 4: Pipeline Architecture
# -------------------------------------------------------------
elif mode == "⚙️ Pipeline Architecture & Specs":
    st.subheader("System Architecture & Implementation Specifications")
    st.markdown(
        """
        ```
        +-------------------------------------------------------------------------------+
        |               End-to-End Transformer Summarization + TTS Pipeline             |
        +-------------------------------------------------------------------------------+
                                                 |
                                                 v
        [ Raw Input Text (5000+ Multi-Domain Documents, Articles, Support Logs) ]
                                                 |
                                                 v [src/preprocessing.py]
        [ Text Normalization, Unicode Cleaning, Contraction Expansion, Token Truncation ]
                                                 |
                                                 v [src/summarization.py]
        [ Hugging Face BART/Pegasus Transformer Abstractive Summarizer (30-40% Ratio) ]
                                                 |
                                                 v [src/tts_pipeline.py]
        [ Neural TTS Engine (Glow-TTS / FastPitch / SpeechT5) + 22050 Hz Normalization ]
                                                 |
                                                 +------------------------------+
                                                 |                              |
                                                 v                              v
        [ Output Speech WAV Audio ]                           [src/evaluation.py]
        [ (data/audio/generated_speeches) ]                   [ ROUGE-1/2/L, Consistency ]
                                                              [ MOS Quality Protocol      ]
        ```
        """
    )
