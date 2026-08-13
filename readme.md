# Music Genre Classification API

> A CNN-powered REST API that identifies music genre from audio uploads in
> real time — enabling automated catalog tagging, playlist generation, and
> content moderation for music streaming and media platforms.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![torchaudio](https://img.shields.io/badge/torchaudio-2.x-purple)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-teal)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-78%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Business Problem

Music streaming platforms ingest hundreds of thousands of tracks per month —
manual genre tagging by human curators costs $2–5 per track and creates
catalog delays of days to weeks before new content becomes searchable.
Automated genre classification enables instant tagging at upload time,
powers recommendation engine inputs, and reduces editorial costs by an
estimated 60–80% for platforms processing large music libraries.

---

## Demo

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@track.wav"
```

**Response:**
```json
{
  "index": 5,
  "genre": "jazz"
}
```

**Supported genres (10 total):**
`blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock`

---

## Results

| Metric      | Score  |
|-------------|--------|
| Train score | 85.61% |
| Test score  | 78.0%  |
| Final loss (epoch 60) | 0.6979 |

Model: `CheckAudio` CNN — Mel spectrogram → `Conv2d(1→16→32)` + BatchNorm →
`AdaptiveAvgPool2d((8,8))` → `Linear(2048→128)` → `Linear(128→10)`, trained
60 epochs with SpecAugment + Gaussian noise augmentation on train split.

Baseline (random classifier, 10 genres): Accuracy = 10%
↑ +68 points over baseline

> Note: State-of-the-art on this benchmark (transformer-based audio models)
> reaches ~90%+. This model reaches 78% test accuracy trained from scratch
> on 1,000 tracks — a solid result given the small dataset and lightweight
> architecture (no pretrained weights).

---

## Dataset

- **Source:** GTZAN Music Genre Collection (Kaggle:
  `andradaolteanu/gtzan-dataset-music-genre-classification`)
- **Size:** 1,000 audio tracks × ~30 seconds each, 22050 Hz mono WAV
- **Features:** Raw waveform → Mel spectrogram (`n_fft=2048`, `hop_length=512`,
  64 mel bins) → `AmplitudeToDB` → truncated/padded to 1280 time frames,
  treated as a single-channel 2D image
- **Class balance:** Perfectly balanced — exactly 100 tracks per genre
- **Split:** 80/20 train/test via `random_split` with fixed seed
  (`manual_seed(42)`), applied identically to both the augmented (train)
  and clean (test) versions of the dataset so augmentation never leaks
  into evaluation

---

## Approach

1. **Data Loading** — Downloaded via `kagglehub`; custom `GTZAN` `Dataset`
   class walks the genre directory tree and collects `(path, genre)` pairs,
   skipping one known corrupted file (`jazz.00054.wav`) that fails to load
2. **Feature Extraction** — Raw waveform → mono → `MelSpectrogram`
   (sample_rate=22050, n_fft=2048, hop_length=512, n_mels=64) →
   `AmplitudeToDB`; output truncated or zero-padded to `max_len=1280`
   frames for fixed-size input
3. **Sample Rate Normalization** — Automatic `Resample` to 22050 Hz for
   any non-standard source files; applied consistently in both training
   pipeline and inference API
4. **Augmentation (train only)** — `FrequencyMasking(15)` +
   `TimeMasking(35)` (SpecAugment) plus light Gaussian noise
   (`σ=0.005`) added directly to the spectrogram
5. **Model Architecture** — Lightweight CNN:
   `Conv2d(1→16)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2)` →
   `Conv2d(16→32)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2)` →
   `AdaptiveAvgPool2d((8,8))` → `Flatten` →
   `Linear(2048→128)` + `ReLU` + `Dropout(0.5)` → `Linear(128→10)`
6. **Training** — 60 epochs, Adam (lr=0.001, weight_decay=1e-4),
   CrossEntropyLoss, `batch_size=32`, GPU-accelerated when available
7. **Inference API** — FastAPI `/predict` endpoint; `soundfile` reads
   uploaded bytes directly from memory via `io.BytesIO`; full
   preprocessing pipeline shared via the `change_audio()` helper

---

## Key Challenges & Solutions

**Corrupted WAV file breaking the training loop**
One file in the dataset (`jazz.00054.wav`) fails to load via `torchaudio`
and would crash the epoch mid-run → added an explicit skip for this path
in the `Dataset.__init__` file-collection step → stable training across
all 60 epochs with no interruptions. This is a targeted fix for a known
file rather than general corruption-detection logic.

**Variable-length audio tracks breaking batch tensor stacking**
Music tracks vary slightly in duration (28–31 seconds), producing spectrograms
of different time-axis lengths — standard `DataLoader` collation fails with
mismatched tensor sizes → added deterministic truncation
(`spec[:, :max_len]`) and zero-padding (`F.pad(spec, (0, pad_len))`) to
normalize all spectrograms to exactly `[64, 1280]` → consistent batching
across all 1,000 tracks with no shape errors.

**Two audio backends in the codebase (FastAPI vs Streamlit)**
The FastAPI inference path reads uploaded bytes via `soundfile`, while an
alternate Streamlit interface (currently commented out in `main.py`) uses
`librosa.load` — different backends can normalize audio slightly
differently for the same input → both paths route through the same
`change_audio()` preprocessing helper to keep the transform logic
identical. The Streamlit path hasn't been exercised in production, so this
is a design safeguard rather than a verified fix for an observed
discrepancy.

---

## Tech Stack

| Category      | Tools                                     |
|---------------|--------------------------------------------|
| Language      | Python 3.11                                |
| ML            | PyTorch, torchaudio                        |
| Audio         | soundfile, librosa, torchaudio.transforms  |
| API           | FastAPI, Uvicorn                            |
| Data          | KaggleHub, pandas                          |
| Deploy        | FastAPI (local / cloud)                    |

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/your-username/music-genre-classifier
cd GTZAN_DatasetMusicGenreClassification
pip install -r requirements.txt
```

```bash
# 2. Train the model
# Run GTZAN_DatasetMusicGenreClassification.ipynb (Google Colab, GPU recommended)
# → saves model_CheckAudio_GTZAN_DatasetMusicGenreClassification.pth
#   + labels_GTZAN_DatasetMusicGenreClassification.pth
```

```bash
# 3. Launch the API
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

---

## Business Impact

- ↓ ~75% reduction in manual genre tagging costs for music catalog teams
  vs human editorial review at scale (estimated)
- ↑ 78% automated genre classification accuracy across 10 categories —
  sufficient for first-pass catalog organization and recommendation
  engine inputs
- ↓ ~90% reduction in time-to-catalog for new track uploads vs manual
  tagging workflows (from days to milliseconds per track) (estimated)
- ↑ REST API architecture integrates directly into upload pipelines —
  genre tag returned synchronously at the moment of file submission
- ↑ Retrainable on proprietary sub-genre taxonomies (e.g. lo-fi, trap,
  ambient) by swapping the dataset directory — no architecture changes

---

## Структура репозитория
```
ml_GTZAN_DatasetMusicGenreClassification/
├── .gitignore
├── readme.md
├── requirements.txt
└── GTZAN_DatasetMusicGenreClassification/
    ├── GTZAN_DatasetMusicGenreClassification.ipynb   # обучение (единственный источник train-кода)
    ├── labels_GTZAN_DatasetMusicGenreClassification.pth
    ├── main.py                                        # FastAPI + Streamlit (закомментирован)
    ├── model_CheckAudio_GTZAN_DatasetMusicGenreClassification.pth
    └── test audio/
        ├── classical.wav
        ├── disco.wav
        └── pop.wav
```
---
