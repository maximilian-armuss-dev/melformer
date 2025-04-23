# 🎶 Melformer  
🚧 **Heavily Under Construction** – This project is in early development. Expect broken things, half-finished thoughts, and lots of experimentation.

## 📌 Overview

**Melformer** is a research-driven music generation project that combines the representational power of a **VQ-VAE** with the sequential modeling capabilities of a **transformer**.

Instead of working with raw audio directly, Melformer introduces a custom audio format (`.tiaf`) that stores a fixed number of samples per **beat**—not per second. This beat-aligned representation allows the model to treat audio like a sequence of tokens, similar to words in NLP.

## 🧠 Ideas & Theory

### 🎼 The `.tiaf` Format
Traditional `.wav` files store **samples per second (e.g. 44,100Hz)**. This causes token misalignment when tempo changes.

`.tiaf` redefines time:  
- Stores a 40k samples **per beat**
- BPM-independent representation
- Makes musical phrases directly tokenizable

Each **beat becomes a token**, forming a clean sequence ideal for transformers.

### Drawbacks

40k samples per token is a lot of data → **training and inference would take ages!**

### 🧱 Model Architecture

In order to save computations while keeping the most relevant features of the original audio data, we leverage the power of a **VQ-VAE** as follows:

![Model Architecture](melformer.svg)

### 🎓 Training Plan

1. **Train VQ-VAE** on `.tiaf` audio until convergence  
2. **Freeze encoder and decoder**  
3. Insert **transformer** between them  
4. Train transformer **autoregressively**  

## 📁 Project Structure

    melformer/
    │
    ├── configs/
    │   └── transformer.json                # Config file for Transformer model
    │
    ├── data/
    │   ├── test_in/                        # Raw .wav files for testing
    │   │   └── [example_loops].wav
    │   └── test_out/                       # Placeholder for processed output
    │
    ├── src/
    │   ├── audio_classes/
    │   │   ├── tiaf.py                     # Convert .wav <-> .tiaf format
    │   │   └── wav.py                      # WAV loading & handling utilities
    │   │
    │   ├── dataset/
    │   │   └── tokenicer_dataset.py        # Dataset class for tokenized audio
    │   │
    │   ├── melformer/
    │   │   ├── encoder/
    │   │   │   ├── encoder.py            
    │   │   │   └── encoder_blocks.py
    │   │   │
    │   │   ├── decoder/
    │   │   │   ├── decoder.py
    │   │   │   └── decoder_blocks.py
    │   │   │
    │   │   └── transformer/
    │   │       ├── classifier.py           # Final linear layer
    │   │       ├── gqa.py                  # Grouped-Query attention
    │   │       ├── kvcache.py              
    │   │       ├── mlp.py                  # Multi-Layer-Perceptron (SwiGLU feedforward network)
    │   │       ├── rms_norm.py             # Root-Mean-Square norm
    │   │       ├── rope.py                 # Rotary positional embeddings
    │   │       ├── transformer_block.py    # Transformer modules with LLaMa-2 architecture
    │   │       └── transformer_model.py    # Overall model
    │   │
    │   └── util/
    │       ├── config_loader.py            # Loads and parses config files
    │       ├── fft_test.py                 # Test for FFT operations
    │       └── stft.py                     # Short-time Fourier Transform tools
    │
    ├── .gitignore
    ├── environment.yml
    └── README.md

## 🛠️ Setup

Coming soon – once things are working.  

## 📬 Contact

If you're curious, want to collaborate, or just geek out over music + AI, feel free to reach out!
