from collections import Counter
import json
import wikipedia as wiki
import os

VOCAB_PATH = './vocab.json'
TRAINING_PATH = './training-data'

def read_training_data(file_path=TRAINING_PATH):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_vocab(file_path=VOCAB_PATH):
    """Reads the existing vocabulary from the given file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        return []

def write_vocab(data, file_path=VOCAB_PATH):
    """Writes vocabulary data to the JSON file in sorted order."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(sorted(data, key=len), file, indent=4)
    except (TypeError, ValueError) as err:
        raise RuntimeError(f"Failed to write vocabulary: {err}")

def merge_corpus(corpus, merged_pair):
    """Merges adjacent tokens in the corpus based on the target pair."""
    skip = False
    merged_corpus = []
    i = 0
    while i < len(corpus):
        if i < len(corpus) - 1 and (corpus[i] + corpus[i + 1]) == merged_pair:
            merged_corpus.append(merged_pair)
            i += 2  # Skip the next token as it's merged
        else:
            merged_corpus.append(corpus[i])
            i += 1

    return merged_corpus

def tokenize(word):
    """Splits a word into known vocabulary tokens."""
    vocab = set(read_vocab())
    subwords = []

    while word:
        for i in range(len(word), 0, -1):
            subword = word[:i]
            if subword in vocab:
                subwords.append(subword)
                word = word[i:]
                break
        else:
            subwords.append("<UNK>")
            break

    return subwords


def update_vocab(text, iterations):
    """Creates or updates vocabulary based on the given text over multiple iterations."""
    text = text.replace("\n", " ")
    if read_vocab() == []:
        vocab = set(text)
    else:
        vocab = set(read_vocab())

    corpus = list(text)
    
    for _ in range(iterations):
        pairs = Counter((corpus[i], corpus[i + 1]) for i in range(len(corpus) - 1))
        if not pairs:
            break
        
        most_common_pair = max(pairs, key=pairs.get)
        merged_token = "".join(most_common_pair)
        
        corpus = merge_corpus(corpus, merged_token)
        vocab.add(merged_token)
        write_vocab(list(vocab))

        print(len(corpus))

    
    return vocab


if __name__ == "__main__":
    for file in os.listdir(TRAINING_PATH):
        print("PROCESSING: " + file)
        data = read_training_data(TRAINING_PATH + '/' + file)

        update_vocab(data, 1000)