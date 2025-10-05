# LASPA Implementation: Language Agnostic Speaker Disentanglement

A PyTorch implementation of language-agnostic speaker recognition using cross-feature prefix tuning, inspired by the LASPA (Language Agnostic Speaker Disentanglement with Prefix-Tuned Cross-Attention) research paper.

## Overview

This implementation addresses the challenge of speaker recognition in multilingual settings by disentangling speaker identity from linguistic features. The model learns to separate speaker-specific characteristics (vocal tract, prosody, timbre) from language-dependent features (phonetic patterns, intonation), enabling robust cross-lingual speaker recognition.

## Architecture

### Core Components

1. **Dual Encoder System**
   - **Speaker Encoder**: Extracts 192-dimensional speaker embeddings using pre-trained SpeechBrain ECAPA-TDNN
   - **Language Encoder**: Generates 256-dimensional language embeddings using VoxLingua107 ECAPA-TDNN

2. **Cross-Feature Prefix Tuners**
   - Bidirectional attention mechanism for feature fusion
   - Speaker-to-Language and Language-to-Speaker cross-attention
   - Learnable prefix parameters for enhanced feature interaction

3. **Decoder Architecture**
   - LSTM-based sequential modeling
   - Multi-layer perceptron for mel-spectrogram reconstruction
   - Layer normalization for stable training

### Key Differences from Original Paper

While inspired by the LASPA paper, this implementation includes several modifications:

- **Encoder Architecture**: Uses pre-trained SpeechBrain models instead of training from scratch
- **Decoder Design**: Implements LSTM-based decoder with FC upsampling layers
- **Dataset**: Trained on Kathbath (Sanskrit) instead of VoxCeleb2
- **Training Strategy**: Simplified training loop with gradient clipping and early stopping
- **Loss Weighting**: Uses equal weights (α=β=γ=δ=1.0) with learnable parameters

## Loss Functions

The model optimizes a multi-objective loss function:

```
L_total = α·L_MSE + β·L_MAPC + γ·L_AAM + δ·L_NLL
```

- **L_MSE**: Mean Squared Error for mel-spectrogram reconstruction
- **L_MAPC**: Mean Absolute Pearson Correlation for disentanglement (minimizes correlation between speaker and language embeddings)
- **L_AAM**: Additive Angular Margin Softmax for speaker classification
- **L_NLL**: Negative Log Likelihood for language classification

## Installation

```bash
# Install required packages
pip install datasets==3.6.0 \
            sentence-transformers==4.1.0 \
            soundfile==0.13.1 \
            speechbrain==1.0.3 \
            torchaudio==2.6.0 \
            transformers==4.52.4 \
            torchcodec

# Additional dependencies
pip install matplotlib scikit-learn
```

## Dataset

The implementation uses the **Kathbath** dataset from ai4bharat:
- Multi-lingual Indian language speech dataset
- Default configuration: Sanskrit language
- 182 unique speakers in training set
- 26,840 training samples
- 16kHz sampling rate

## Usage

### Training

```python
# Basic training
python train.py

# The script will:
# 1. Load pre-trained encoders
# 2. Initialize LASPA model
# 3. Setup data loaders
# 4. Train for 5 epochs
# 5. Save checkpoints and final model
```

### Model Configuration

```python
model = LASPA(
    proj_dim=192,           # Projection dimension
    num_heads=4,            # Attention heads
    prefix_len=5,           # Prefix sequence length
    n_mels=80,              # Mel-spectrogram bins
    hidden_dim=512,         # LSTM hidden dimension
    num_speakers=182,       # Number of speakers
    num_languages=1         # Number of languages
)
```

### Training Parameters

- **Optimizer**: AdamW (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=1)
- **Batch Size**: 4
- **Epochs**: 5
- **Gradient Clipping**: max_norm=1.0

## Model Details

### Feature Extraction Pipeline

```
Audio Waveform (16kHz)
    ↓
Mel-Spectrogram (80 bins, 25ms window, 10ms hop)
    ↓
Speaker Encoder (192-dim) + Language Encoder (256-dim)
    ↓
Projection Layers (both → 192-dim)
    ↓
Cross-Feature Prefix Tuners (bidirectional attention)
    ↓
Concatenated Features (384-dim)
    ↓
LSTM Decoder → Reconstructed Mel-Spectrogram
```

### Prefix-Tuned Cross-Attention

The cross-attention mechanism uses learnable prefix parameters:

```python
# Language-to-Speaker flow
Q = language_embedding
K = [prefix_k, speaker_embedding]
V = [prefix_v, speaker_embedding]

# Speaker-to-Language flow
Q = speaker_embedding
K = [prefix_k, language_embedding]
V = [prefix_v, language_embedding]
```

## Training Results

Sample training metrics (Epoch 5):
- **Average Loss**: 96.72
- **MSE Loss**: 88.43 (reconstruction quality)
- **MAPC Loss**: 0.036 (disentanglement quality)
- **AAM Loss**: 8.26 (speaker classification)
- **NLL Loss**: 0.00 (single language scenario)

## Checkpoints

The training script saves:
- Epoch checkpoints: `laspa_checkpoint_epoch_{N}.pth`
- Final model: `laspa_final_model.pth`

Each checkpoint contains:
```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_fn_state_dict': loss_fn.state_dict(),
    'avg_loss': average_loss
}
```

## Visualization Tools

```python
# Visualize speaker and language embeddings
visualize_embeddings(model, dataloader, max_batches=5)

# Compare ground truth vs reconstructed mel-spectrograms
plot_mel_comparison(mel_true, mel_recon, idx=0)
```

## Model Parameters

- **Total Parameters**: ~5.1M
- **Trainable Parameters**: ~5.1M
- **Prefix Tuner Parameters**: ~1.16% of total

## Limitations & Future Work

**Current Limitations:**
- Trained only on Sanskrit (single language)
- Limited to 182 speakers
- No multi-GPU training implementation
- No inference-only mode optimization

**Potential Improvements:**
- Multi-lingual training on diverse datasets
- Support for streaming inference
- Model quantization for edge deployment
- Integration with speaker diarization pipelines

## Citation

If you use this implementation, please cite the original LASPA paper:

```bibtex
@article{menon2025laspa,
  title={LASPA: Language Agnostic Speaker Disentanglement with Prefix-Tuned Cross-Attention},
  author={Menon, Aditya Srinivas and Gohil, Raj Prakash and Tripathi, Kumud and Wasnik, Pankaj},
  journal={arXiv preprint arXiv:2506.02083},
  year={2025}
}
```

## License

This implementation is provided for research purposes. Please refer to the original LASPA paper and dataset licenses for usage restrictions.

## Acknowledgments

- SpeechBrain team for pre-trained models
- ai4bharat for the Kathbath dataset
- Original LASPA authors for the research framework

---

**Note**: This is a research implementation and may require tuning for production use. The model architecture and training procedure differ from the original paper in several aspects as noted above.
