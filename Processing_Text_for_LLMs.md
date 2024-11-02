# Working with Text Data for Large Language Models (LLMs)

This guide provides a step-by-step overview of preparing text data for training large language models, covering text tokenization, embeddings, and data sampling strategies essential for LLMs.

---

## 1. Introduction
Before we can train a large language model (LLM), we need to transform raw text into a format suitable for neural network processing. This process includes:
- **Tokenizing** text into individual words or subwords.
- **Encoding** tokens as numeric vectors.
- **Sampling** data for training tasks like next-word prediction.

### Mind Mapping - Key Concepts:
- **Tokenization**
  - Basic tokenization
  - Byte Pair Encoding (BPE)
- **Encoding** 
  - Token IDs
  - Special tokens
- **Sampling**
  - Sliding window approach

---

## 2. Preparing Text for LLM Training

### 2.1 Tokenizing Text
Tokenization breaks text into tokens (words or subwords) compatible with neural network models. Methods include:
- **Basic Tokenization**: Splitting on whitespace and punctuation.
- **Byte Pair Encoding (BPE)**: An advanced technique that handles unknown words by breaking them into subword units, widely used in GPT models.

#### Code Example: Basic Tokenization with `re.split`

```python
import re

text = "Hello, world. This, is a test."
tokens = re.split(r'([,.:;?_!"()']|\s)', text)
tokens = [token.strip() for token in tokens if token.strip()]
print(tokens)
# Output: ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

#### Code Example: Using Byte Pair Encoding (BPE) with `tiktoken`

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, world. This is a test."
token_ids = tokenizer.encode(text)
print(token_ids)
```

### 2.2 Converting Tokens to Token IDs
To enable processing by an LLM, each token is mapped to a unique integer:
- **Building a Vocabulary**: Unique tokens are aggregated, allowing conversion between words and token IDs.
- **Special Tokens**: Special tokens like `<|unk|>` (unknown) and `<|endoftext|>` (end of text) provide context to the model.

#### Code Example: Creating a Vocabulary and Mapping to Token IDs

```python
vocab = {token: idx for idx, token in enumerate(set(tokens))}
token_ids = [vocab[token] for token in tokens]
print(vocab)
print(token_ids)
```

---

### Mind Mapping - Preparing Text
1. **Tokenizing Text** 
   - Basic Tokenization
   - BPE Tokenization
2. **Converting Tokens**
   - Vocabulary building
   - Adding special tokens

---

## 3. Creating Embeddings

### 3.1 Word Embeddings
LLMs need a numeric representation for each token (embedding) to process text data:
- **Embedding Layer**: A neural network layer that maps each token ID to a continuous vector representation.
- **Optimizing Embeddings**: Rather than using fixed embeddings, LLMs learn embeddings specific to the task during training.

#### Code Example: Creating Embeddings with PyTorch

```python
import torch

vocab_size = len(vocab)
embedding_dim = 50  # Example dimension size
embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

# Convert token IDs to embeddings
token_embeddings = embedding_layer(torch.tensor(token_ids))
print(token_embeddings.shape)
```

### 3.2 Adding Positional Embeddings
Token embeddings alone do not account for the order of tokens in a sequence. We use:
- **Absolute Positional Embeddings**: Added to each token embedding to encode the position of tokens.
- **Relative Positional Embeddings**: Encode distance rather than exact positions, useful for variable-length sequences.

#### Code Example: Adding Positional Embeddings

```python
context_length = len(token_ids)
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
```

---

### Mind Mapping - Embeddings
1. **Token Embeddings**
   - Embedding layer
2. **Positional Embeddings**
   - Absolute
   - Relative

---

## 4. Data Sampling with a Sliding Window
Sampling data pairs (input-target) is essential for training:
- **Sliding Window Approach**: Each input context shifts by one position, creating overlapping input-target pairs for the next-word prediction task.

### 4.1 Implementing a Data Loader
A data loader fetches data in batches, where each batch consists of input and target pairs.

#### Code Example: Data Loader with Sliding Window

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, max_length=5):
        self.token_ids = token_ids
        self.max_length = max_length

    def __len__(self):
        return len(self.token_ids) - self.max_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.token_ids[idx:idx+self.max_length]),
            torch.tensor(self.token_ids[idx+1:idx+self.max_length+1])
        )

# Initialize dataset and dataloader
dataset = TextDataset(token_ids)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example of fetching a batch
for inputs, targets in dataloader:
    print("Inputs:", inputs)
    print("Targets:", targets)
    break
```

---

### Mind Mapping - Data Sampling
1. **Sliding Window Approach**
   - Creates overlapping input-target pairs
2. **Data Loader**
   - Fetches input and target pairs in batches

---

## 5. Putting It All Together
Once token IDs and embeddings are created, positional embeddings are added to the token embeddings. This combined embedding vector can then be processed by the LLM layers.
