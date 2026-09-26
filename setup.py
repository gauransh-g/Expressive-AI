from setuptools import setup, find_packages

setup(
    name="neural-tts-summarization",
    version="1.0.0",
    description="End-to-End Transformer Text Summarization + Neural TTS Pipeline",
    author="AI Engineering Team",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "rouge-score>=0.1.2",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "tts-summarizer=main:main",
        ],
    },
)
