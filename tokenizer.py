from collections import Counter, defaultdict
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
            json.dump(data, file, indent=4)
    except (TypeError, ValueError) as err:
        raise RuntimeError(f"Failed to write vocabulary: {err}")

def init_vocab(text):
    '''Creates a dictionary that represents the Count of each Character in the given Text'''
    char_vocab = defaultdict(int)

    for char in list(text):
        char_vocab[char] += 1

    return char_vocab

def get_most_common_pair(corpus):
    pairs = Counter(zip(corpus, corpus[1:]))
    if not pairs:
        return None
    
    print(pairs.most_common(1))
    most_common_pair = pairs.most_common(1)[0]
    return ("".join(most_common_pair[0]), most_common_pair[1])

def merge_corpus(corpus, merged_pair):
    """Merges adjacent tokens in the corpus based on the target pair."""
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

def train_vocab(iterations):
    for file in os.listdir(TRAINING_PATH):
        if file.endswith('.txt'):
            print("PROCESSING: " + file)
            text = read_training_data(TRAINING_PATH + '/' + file).replace("\n", " ")

            corpus = list(text)
            existing_vocab = defaultdict(int, read_vocab())
            vocab = existing_vocab if existing_vocab else init_vocab(text)
            
            for _ in range(iterations):
                most_common_pair = get_most_common_pair(corpus)
                
                corpus = merge_corpus(corpus, most_common_pair[0])
                vocab[most_common_pair[0]] = most_common_pair[1]

            write_vocab(dict(sorted(vocab.items(), key=lambda item: item[1])))

def tokenize(word):
    """Splits a word into known vocabulary tokens."""
    vocab = defaultdict(int, read_vocab())
    subwords = []

    while word:
        for i in range(len(word), 0, -1):
            subword = word[:i]
            if subword in vocab.keys():
                subwords.append(subword)
                word = word[i:]
                break
        else:
            subwords.append("<UNK>")
            break

    return subwords

if __name__ == "__main__":
    train_vocab(2)