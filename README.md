# Project Setup

This project uses a set of Python dependencies listed in `requirements.txt`. To ensure a consistent development environment, follow the steps below to install them.

## Prerequisites

- Python 3.7 or higher
- `pip` package manager

## Installation

1. **Clone the repository (if applicable):**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Execution
### BPE Tokenizer
```bash
cd bpe-tokenizer
python3 bpe-tokenizer.py
```

#### Training
1. Place the training texts as `.txt` files into the `./bpe-tokenizer/training-data` folder.
2. Uncomment the `train_vocab` function and edit the iterations per file if needed

#### Tokenizing Text
To tokenize a Text using a vocabulary use the `tokenize` function and pass the text as well as the relative path to the desired vocabulary as funciton parameters
