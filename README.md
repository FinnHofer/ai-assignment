# Project Setup

This project uses a set of Python dependencies listed in `requirements.txt`. To ensure a consistent development environment, follow the steps below to install them.

## Prerequisites

- Python 3.7 or higher
- `pip` package manager

## Installation

1. **Clone the repository (if applicable):**

   ```bash
   git clone https://github.com/FinnHofer/ai-assignment.git
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

### Word Calculator
```bash
cd word-calculator
python3 word-calculator.py
```

For the word calculator to work you have to download the GloVe-Embeddings from the Stanford Website:
https://nlp.stanford.edu/data/glove.6B.zip

Then you want to extract the zip and move the `glove.6B.50d.txt` in the `./word-calculator folder`.\
You can choose a bigger Embedding as well. Just update the filename, set in the python script.

### TextRank
```bash
cd textrank
python3 textrank.py
```

To extract the most important sentence out of a text using TF-IDF and BERT, paste your text into the `text.txt` file or create your own text file and change the path in the code.

#### Plot with Matplotlib
To Plot Graphes just uncomment the according `plot_graph` function