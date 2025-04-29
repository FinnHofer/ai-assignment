from collections import Counter, defaultdict
import json
import os

VOCAB_PATH = './output'
TRAINING_PATH = './training-data'

def read_training_data(file_path):
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
        with open(folder_path+'/'+file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    except (TypeError, ValueError) as err:
        raise RuntimeError(f'Failed to write vocabulary: {err}')

def init_vocab(text):
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
            text = read_training_data(TRAINING_PATH + '/' + file)
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

def tokenize(text, vocab_path):
    vocab = defaultdict(int, read_vocab(vocab_path))
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
    #train_vocab(600)
    print(tokenize('Am Abflugtag fahren Herr und Frau Müller mit ihren beiden Kindern im Taxi zum Flughafen. Dort warten schon viele Urlauber. Alle wollen nach Mallorca fliegen. Familie Müller hat viel Gepäck dabei: drei große Koffer und zwei Taschen. Die Taschen sind Handgepäck. Familie Müller nimmt sie mit in das Flugzeug. Am Flugschalter checkt die Familie ein und erhält ihre Bordkarten. Die Angestellte am Flugschalter erklärt Herrn Müller den Weg zum Flugsteig. Es ist nicht mehr viel Zeit bis zum Abflug. Familie Müller geht durch die Sicherheitskontrolle. Als alle das richtige Gate erreichen, setzen sie sich in den Wartebereich. Kurz darauf wird ihre Flugnummer aufgerufen und Familie Müller steigt in das Flugzeug nach Mallorca.', VOCAB_PATH + '/' + 'training-de-vocab.json'))
    print(tokenize('Am Abflugtag fahren Herr und Frau Müller mit ihren beiden Kindern im Taxi zum Flughafen. Dort warten schon viele Urlauber. Alle wollen nach Mallorca fliegen. Familie Müller hat viel Gepäck dabei: drei große Koffer und zwei Taschen. Die Taschen sind Handgepäck. Familie Müller nimmt sie mit in das Flugzeug. Am Flugschalter checkt die Familie ein und erhält ihre Bordkarten. Die Angestellte am Flugschalter erklärt Herrn Müller den Weg zum Flugsteig. Es ist nicht mehr viel Zeit bis zum Abflug. Familie Müller geht durch die Sicherheitskontrolle. Als alle das richtige Gate erreichen, setzen sie sich in den Wartebereich. Kurz darauf wird ihre Flugnummer aufgerufen und Familie Müller steigt in das Flugzeug nach Mallorca.', VOCAB_PATH + '/' + 'training-de-en-vocab.json'))
    print(tokenize('I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food! My family is very important to me. We do lots of things together. My brothers and I like to go on long walks in the mountains. My sister likes to cook with my grandmother. On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.', VOCAB_PATH + '/' + 'training-en-vocab.json'))
    print(tokenize('I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food! My family is very important to me. We do lots of things together. My brothers and I like to go on long walks in the mountains. My sister likes to cook with my grandmother. On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.', VOCAB_PATH + '/' + 'training-de-en-vocab.json'))