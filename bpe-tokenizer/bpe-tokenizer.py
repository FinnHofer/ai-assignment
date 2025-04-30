from collections import Counter, defaultdict
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOCAB_PATH = PROJECT_ROOT / 'bpe-tokenizer' / 'vocabularies'
TRAINING_PATH = PROJECT_ROOT / 'bpe-tokenizer' / 'training-data'
INPUT_TEXT = PROJECT_ROOT / 'bpe-tokenizer' / 'input-text'

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().replace('\n', ' ')

def read_vocab(file_path=VOCAB_PATH):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(error)
        return []

def write_vocab(data, file_name='vocab.json', folder_path=VOCAB_PATH):
    try:
        with open(folder_path / file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    except (TypeError, ValueError) as err:
        raise RuntimeError(f'Failed to write vocabulary: {err}')

def init_vocab(text):
    vocab = defaultdict(int)

    for char in list(text):
       vocab[char] += 1

    return vocab

def get_most_common_pair(corpus):
    pairs = Counter(zip(corpus, corpus[1:]))
    if not pairs:
        return None
    
    print(pairs.most_common(1))
    most_common_pair = pairs.most_common(1)[0]
    return (''.join(most_common_pair[0]), most_common_pair[1])

def merge_corpus(corpus, merged_pair):
    merged_corpus = []
    i = 0
    while i < len(corpus):
        if i < len(corpus) - 1 and (corpus[i] + corpus[i + 1]) == merged_pair:
            merged_corpus.append(merged_pair)
            i += 2
        else:
            merged_corpus.append(corpus[i])
            i += 1

    return merged_corpus

def train_vocab(iterations):
    for file in os.listdir(TRAINING_PATH):
        if file.endswith('.txt'):
            text = read_text_file(TRAINING_PATH / file)
            print('PROCESSING FILE: ' + file)

            corpus = list(text)
            vocab = init_vocab(text)
            
            for _ in range(iterations):
                most_common_pair = get_most_common_pair(corpus)
                if not most_common_pair:
                    break

                corpus = merge_corpus(corpus, most_common_pair[0])
                vocab[most_common_pair[0]] = most_common_pair[1]

            vocab_file_name = file.replace('.txt', '-vocab.json')
            write_vocab(dict(sorted(vocab.items(), key=lambda item: item[1])), vocab_file_name)

def tokenize(file_name, vocab_name):
    text = read_text_file(INPUT_TEXT / file_name)
    vocab = defaultdict(int, read_vocab(VOCAB_PATH / vocab_name))
    subwords = []

    while text:
        for i in range(len(text), 0, -1):
            subword = text[:i]
            if subword in vocab.keys():
                subwords.append(subword)
                text = text[i:]
                break
        else:
            subwords.append('<UNKNOWN>')
            break
        
    return subwords, len(subwords)

if __name__ == '__main__':
    train_vocab(600)