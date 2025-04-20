# Deep Learning Assignment 2

This repository contains two implementations:

1. **Q1** - Sequence-to-Sequence RNN-based character-level transliteration from Latin to Devanagari.
2. **Q2** - Fine-tuning GPT-2 to generate English song lyrics.

---


##  Q-1) Seq2Seq RNN Transliteration Model (Latin → Devanagari)

This project implements a flexible **character-level sequence-to-sequence (seq2seq)** model for transliteration from Latin characters to Devanagari script using a customizable RNN-based encoder-decoder architecture.

---

### 🗂️ Dataset

Dataset Link: [Dakshina Dataset (Google)](https://github.com/google-research-datasets/dakshina)\
Files used:

- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

### 🚀 Model Architecture

The model includes:
1. **Character Embedding Layer** – Converts input characters into dense vectors of size `m`.
2. **Encoder RNN** – Processes the input sequence using a single-layer or multi-layer RNN, LSTM, or GRU with hidden size `k`.
3. **Decoder RNN** – Initialized with the final encoder state and outputs Devanagari characters one at a time.
4. **Linear + Softmax Layer** – Projects decoder output to vocabulary space.

---

### ⚙️ Customization Parameters

The code is built to easily configure the following:

| Parameter            | Description                                 |
|---------------------|---------------------------------------------|
| `embedding_dim (m)` | Size of character embeddings                |
| `hidden_dim (k)`    | Size of hidden states for encoder/decoder  |
| `cell_type`         | Type of RNN cell: `RNN`, `LSTM`, or `GRU`  |
| `num_layers`        | Number of layers in both encoder/decoder   |
| `vocab_size (V)`    | Vocabulary size (same for source/target)   |
| `seq_length (T)`    | Input/output sequence length               |

---

### 🧮 (a) Total Number of Computations

Assumptions:
- Input embedding size = `m`
- Hidden size = `k`
- Input and output sequence length = `T`
- Using 1-layer encoder and decoder

Each time step (encoder or decoder) involves: O(k² + m·k)

So total computations: T × [Encoder RNN + Decoder RNN] = T × 2 × (k² + m·k)

---

### 📐 (b) Total Number of Parameters

Breakdown:

- **Embedding Layer**: `V × m`
- **Encoder RNN**: `m × k + k × k + k`
- **Decoder RNN**: `k × k + k × k + k`
- **Output Layer**: `k × V + V`

Total Parameters: = V×m + 2×(m×k + k² + k) + k×V + V

---

### ✅ (c) Evaluation Results


- **Best Configuration**:
  - Cell Type: `LSTM`
  - Embedding Size (m): `128`
  - Hidden Size (k): `256`
  - Number of Layers: `1`
- **Test Accuracy**: `94.48%`
---
### 🔍 Sample Predictions

***Sample 1***
- ***Input (Latin)***: a n k  
- ***Target***: `<start>` अ ं क `<end>`  
- ***Predicted***: अंक

***Sample 2**
- ***Input (Latin)***: a n k a  
- ***Target***: `<start>` अ ं क `<end>`  
- ***Predicted***: अंका

***Sample 3***
- ***Input (Latin)***: a n k i t  
- ***Target***: `<start>` अ ं क ि त `<end>`  
- ***Predicted***: अंकित

***Sample 4***
- ***Input (Latin)***: a n a k o n  
- ***Target***: `<start>` अ ं क ो ं `<end>`  
- ***Predicted***: अनाकों

***Sample 5***
- ***Input (Latin)***: a n k h o n  
- ***Target***: `<start>` अ ं क ो ं `<end>`  
- ***Predicted***: अंखों


### 📂 File Structure

```
.
├── README.md
├── main_seq2seq_transliteration.py  # All code for Q1
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```

---

📘 References

- [Keras LSTM Seq2Seq Example](https://keras.io/examples/nlp/lstm_seq2seq/)
- [Machine Learning Mastery - Seq2Seq](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)

---


##  Q-2)🎵 GPT-2 Based Lyrics Generator (Fine-tuning for Song Lyrics)

This project fine-tunes the **GPT-2 language model** to generate English song lyrics using HuggingFace's `transformers` library and publicly available song lyrics datasets.

---

### 📚 Dataset

Dataset : [Poetry and Lyrics Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/poetry)

You can choose the above and preprocess them into a `.txt` file with one lyric line per entry.

---

### 🧠 Model Overview

We use the `GPT-2` transformer model from HuggingFace's `transformers` library. The model is fine-tuned on a custom dataset of song lyrics using causal language modeling.

---
### 🔧 Methodology

- Preprocess the lyrics text and convert it into a suitable format for training
- Use Hugging Face's `GPT2Tokenizer` and `GPT2LMHeadModel`
- Fine-tune GPT-2 on the lyric dataset using the `Trainer` API
- Generate song lyrics using a seed prompt

---
### 🛠️ Tools & Libraries

- Python
- Hugging Face Transformers
- Datasets (🤗)
- PyTorch
- Google Colab (for GPU acceleration)

---

### 🧪 Training Details

- Model: GPT-2
- Epochs: 3
- Max length: 512
- Batch size: 2
- Optimizer: AdamW
- Padding token: `<eos>`

---

### 💬 Sample Output

**Prompt:** `Shine bright like a diamond`

**Generated Lyrics:** `But I don't think it's a 
pretty beautiful diamond or a nice color. 
It's not a really beautiful color but it has 
a lot of its own uniqueness. I think that's the 
beauty of it. So I'm pretty sure it is a beautiful white diamond
but I still think the natural color is not that beautiful. 
You know, it could be a little natural but you could see it quite clearly. 
That's all I know for now.`

---
### 📂 File Structure

```
.
├── README.md
├── gpt2_lyrics_finetune.py
└── /fine_tuned_lyrics_gpt2/  # Saved model
```

---

📘 References

- [GPT-2 Fine-tuning Tutorial](https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a)
- [Poetry Dataset on Kaggle](https://www.kaggle.com/paultimothymooney/poetry)

---









