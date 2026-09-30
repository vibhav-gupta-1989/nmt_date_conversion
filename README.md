# Neural Machine Translation for Date Format Conversion

A PyTorch sequence-to-sequence neural machine translation (NMT) model that converts dates from natural-language formats such as:

```text
September 30, 2006
```

into standardized ISO-style formats:

```text
2006-09-30
```

The project demonstrates an end-to-end NMT pipeline using synthetic data, BPE tokenization, an encoder-decoder GRU architecture, teacher forcing, and GPU-based training.

## Project Overview

The model learns to translate dates between two different representations:

**Input**

```text
April 22, 2019
```

**Output**

```text
2019-04-22
```

The dataset is generated programmatically, so no external dataset is required.

The project covers:

* Synthetic dataset generation
* BPE tokenization using the Hugging Face `tokenizers` library
* PyTorch `DataLoader` and custom batch collation
* Encoder-decoder GRU architecture
* Teacher-forced sequence generation
* Cross-entropy training
* GPU acceleration with CUDA
* Multiclass token-level accuracy evaluation

## Dataset

The training, validation, and test datasets are generated randomly.

| Dataset    | Samples |
| ---------- | ------: |
| Training   |  10,000 |
| Test       |   1,000 |

Dates are generated for years between **1900 and 2025**, with valid days selected according to the number of days in each month.

### Example

```text
Source: September 30, 2006
Target: 2006-09-30
```

The Python `calendar` module is used to ensure that generated dates are valid.

## Tokenization

The project uses a **Byte Pair Encoding (BPE)** tokenizer from the `tokenizers` library.

```python
nmt_tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
```

The tokenizer uses whitespace pre-tokenization and is trained on both source and target dates from the training set.

Special tokens include:

```text
<pad>
<unk>
<s>
</s>
```

The tokenizer has a vocabulary size of up to **10,000 tokens** and a maximum sequence length of **256**.

For example:

```python
nmt_tokenizer.encode("September 30, 2006").ids
```

produces:

```text
[97, 127, 4, 230]
```

## Model Architecture

The model uses an encoder-decoder sequence-to-sequence architecture based on **GRUs**.

```text
                 Source Date
                     │
                     ▼
               BPE Tokenizer
                     │
                     ▼
                Embedding
                     │
                     ▼
              GRU Encoder
                     │
              Hidden State
                     │
                     ▼
              GRU Decoder
                     │
                     ▼
             Linear Projection
                     │
                     ▼
              Output Tokens
                     │
                     ▼
              Target Date
```

### Encoder

The encoder consists of:

* Embedding dimension: `512`
* Hidden dimension: `512`
* GRU layers: `2`

### Decoder

The decoder uses the same embedding layer and consists of:

* Embedding dimension: `512`
* Hidden dimension: `512`
* GRU layers: `2`

The encoder's final hidden state is passed to the decoder to initialize the decoding process.

### Output Layer

The decoder output is projected to the tokenizer vocabulary using a linear layer:

```python
self.output = nn.Linear(hidden_dim, vocab_size)
```

## Training

The model is trained using:

```python
nn.CrossEntropyLoss(ignore_index=0)
```

where token ID `0` corresponds to the padding token.

The optimizer is:

```python
torch.optim.NAdam(nmt_model.parameters())
```

Training configuration:

```text
Batch size: 32
Epochs: 5
Optimizer: NAdam
Loss: Cross Entropy
Device: CUDA
```

During training, the target sequence is shifted to implement teacher forcing:

```text
Input to decoder:  <s> 2006-09-30
Target sequence:       2006-09-30 </s>
```

## Results

The model achieved the following token-level accuracy during training:

| Epoch | Training Accuracy | Training Loss |
| ----: | ----------------: | ------------: |
|     1 |            94.53% |        0.2814 |
|     2 |           100.00% |        0.0012 |
|     3 |           100.00% |        0.0005 |
|     4 |           100.00% |        0.0003 |
|     5 |           100.00% |        0.0002 |

The final evaluation on the generated test set achieved:

```text
Test Accuracy: 100.00%
```

> **Note:** The reported accuracy is token-level multiclass accuracy, not a separate exact-sequence accuracy metric.

## Requirements

The project requires Python with the following packages:

```bash
pip install torch tokenizers torchmetrics
```

CUDA is required by the current notebook implementation because the model and evaluation metric are explicitly moved to:

```python
.to("cuda")
```

## Running the Project

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the dependencies:

```bash
pip install torch tokenizers torchmetrics
```

3. Make sure a CUDA-compatible PyTorch installation is available.

4. Open the notebook:

```text
code.ipynb
```

5. Run the cells sequentially.

The notebook will:

1. Generate the training dataset
2. Generate the validation dataset
3. Generate the test dataset
4. Train the BPE tokenizer
5. Create the data loaders
6. Build the GRU encoder-decoder model
7. Train the model for 5 epochs
8. Evaluate the trained model on the test set

## Technologies Used

* **Python**
* **PyTorch**
* **GRU / RNN**
* **Neural Machine Translation**
* **BPE Tokenization**
* **Sequence-to-Sequence Learning**
* **CUDA / GPU Computing**
* **TorchMetrics**
* **Hugging Face `tokenizers`**

## Key Learning Outcomes

This project demonstrates practical implementation of:

* Sequence-to-sequence neural networks
* Encoder-decoder architectures
* Recurrent neural networks with GRUs
* Tokenization of structured text
* BPE tokenization
* Padding and attention masks
* Packed sequences using `pack_padded_sequence`
* Teacher forcing
* Multiclass cross-entropy loss
* GPU-based model training
* Sequence translation using PyTorch

## Limitations

This project uses synthetically generated dates with a fixed input and output format. Therefore, it is primarily intended as a demonstration of **sequence-to-sequence learning and neural machine translation**, rather than as a production date-parsing system.

A production system could use deterministic date parsing instead, which would be more appropriate when the input formats and transformation rules are known in advance.

## Project Structure

```text
.
├── code.ipynb
└── README.md
```

## Example

```text
Input:
September 30, 2006

Output:
2006-09-30
```

---

**Built with PyTorch to explore neural machine translation, sequence-to-sequence modeling, and GRU-based encoder-decoder architectures.**
