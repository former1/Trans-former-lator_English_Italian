# 🇬🇧 → 🇮🇹 English to Italian Neural Machine Translator

A Transformer-based sequence-to-sequence model trained from scratch to translate English text into Italian. Built with PyTorch and HuggingFace Accelerate.

---

## Overview

This project implements a classic encoder-decoder Transformer architecture (Vaswani et al., 2017) trained on the [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) English-Italian parallel corpus. Everything is built from scratch — the model architecture, tokenizer training, data pipeline, and training loop.

---

## Architecture

| Component | Details |
|---|---|
| Model | Encoder-Decoder Transformer |
| Parameters | ~92.5M |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| FFN Hidden Size | 2048 (4x ratio) |
| Max Sequence Length | 512 tokens |
| Positional Encoding | Fixed sinusoidal |

---

## Tokenization

- **English (source):** [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) WordPiece tokenizer — vocab size 30,522
- **Italian (target):** Custom WordPiece tokenizer trained from scratch on the Italian side of the corpus — vocab size 32,000

The Italian tokenizer is trained using the HuggingFace `tokenizers` library with:
- NFC normalization + lowercasing
- Whitespace pre-tokenization
- Special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`

---

## Dataset

**Source:** [Helsinki-NLP/opus-100 (en-it)](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

| Split | Samples |
|---|---|
| Train | 895,853 |
| Validation | 1,937 |
| Test | 1,952 |

The dataset is pre-tokenized and saved to disk before training for efficiency.

---

## Project Structure

```
transformer_lator/
├── model.py              # Transformer architecture (encoder, decoder, attention, FFN)
├── train.py              # Training loop with gradient accumulation & checkpointing
├── data.py               # DataLoader collation with padding and masking
├── tokenizer.py          # Italian WordPiece tokenizer (training + inference)
├── prepare_data.py       # Dataset download and pre-tokenization pipeline
└── trained_tokenizer/
    └── italian_wp.json   # Saved Italian tokenizer vocab
```

---

## Training

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Betas | (0.9, 0.98) |
| Epsilon | 1e-6 |
| Weight Decay | 0.001 |
| LR Scheduler | Cosine with warmup |
| Warmup Steps | 2,000 |
| Total Steps | 50,000 |
| Batch Size | 4 (effective, with gradient accumulation) |
| Gradient Accumulation | 2 steps |
| Gradient Clipping | 1.0 |

**To prepare the dataset:**
```bash
python prepare_data.py \
  --path_to_save /path/to/save \
  --max_length 512 \
  --min_length 5
```

**To train:**
```bash
accelerate launch train.py
```

**To resume from checkpoint:**
```python
# In train.py, set:
resume_from_checkpoint = "checkpoint_5000"  # replace with your checkpoint name
```

---

## Inference

The model uses greedy decoding at inference time. Example:

```python
from model import Transformer, TransformerConfig
from tokenizer import ItalianTokenizer
from transformers import AutoTokenizer
import torch

# Load tokenizers
src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
tgt_tokenizer = ItalianTokenizer("trained_tokenizer/italian_wp.json")

# Load model
config = TransformerConfig()
model = Transformer(config)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

# Translate
src_ids = torch.tensor(src_tokenizer("I love Italy")["input_ids"]).unsqueeze(0)
output_ids = model.inference(
    src_ids,
    tgt_start_id=tgt_tokenizer.special_tokens_dict["[BOS]"],
    tgt_end_id=tgt_tokenizer.special_tokens_dict["[EOS]"]
)
print(tgt_tokenizer.decode(output_ids, skip_special_tokens=True))
# → "amo l'italia"
```

---

## Requirements

```
torch
transformers
datasets
accelerate
tokenizers
numpy
tqdm
wandb  # optional, for experiment tracking
```

Install with:
```bash
pip install torch transformers datasets accelerate tokenizers numpy tqdm wandb
```

---

## References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Helsinki-NLP/opus-100 dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100)


## Results

After training for **50,000 steps**, the model achieved the following evaluation performance on the **OPUS-100 English–Italian test set**.

### BLEU Score

**BLEU: 21.76**

---

### Sample Translations

**[1]**

|                  |                                          |
| ---------------- | ---------------------------------------- |
| **Input**        | I spoke like a real public servant.      |
| **Model Output** | ho parlato come un vero servo pubblico.  |
| **Reference**    | parli come una vera dipendente pubblica. |

---

**[2]**

|                  |                                                                            |
| ---------------- | -------------------------------------------------------------------------- |
| **Input**        | I wrote you a letter when I was at the academy.                            |
| **Model Output** | ti ho scritto una lettera quando ero all ' accademia.                      |
| **Reference**    | e ' un piacere conoscerla. le... le ho scritto quando ero all ' accademia. |

---

**[3]**

|                  |                                          |
| ---------------- | ---------------------------------------- |
| **Input**        | The bomber sneaks into the shop...       |
| **Model Output** | - gli scopiti nel negozio...             |
| **Reference**    | il bombarolo si intrufola nel negozio... |

---

### Observations

* The model captures **core semantic structure** in most sentences.
* Minor issues appear in:

  * **Morphological agreement** (gender/number)
  * **Word choice ambiguity**
  * **Tokenization artifacts** (`'` spacing)
* Despite these limitations, the model produces **grammatically plausible Italian translations** for many inputs.

These results demonstrate that a **from-scratch Transformer with ~92M parameters** can learn meaningful translation behavior using the OPUS-100 dataset and a modest training setup.

---
