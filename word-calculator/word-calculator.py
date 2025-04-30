import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import re
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GLOVE_PATH = PROJECT_ROOT / 'word-calculator' / 'glove.6B.50d.txt'
CONVERTED_PATH = PROJECT_ROOT / 'word-calculator' / 'glove.6B.50d.word2vec.txt'

def load_glove_model(glove_path_txt, converted_path):
    if not os.path.exists(converted_path):
        glove2word2vec(glove_path_txt, converted_path)
    model = KeyedVectors.load_word2vec_format(converted_path, binary=False)
    return model

def parse_input(usr_input):
    calc = usr_input.replace(" ", "")
    if not calc.startswith(('+', '-')):
        calc = '+' + calc

    # Split into list of tuples: [('+', 'king'), ('-', 'man'), ('+', 'woman')]
    tokens = re.findall(r'([+-])([^\+-]+)', calc)

    additions = [word for sign, word in tokens if sign == '+']
    subtractions = [word for sign, word in tokens if sign == '-']
    all_words = additions + subtractions

    return additions, subtractions

## GLOVE FUNCTIONS
def embedding_calc_glove(glove, additions, subtractions):
    words = additions + subtractions
    glove_emb = {word: glove[word] for word in words if word in glove}
    missing = [word for word in words if word not in glove]

    if missing:
        print(f"{', '.join(missing)} are not in the GloVe model")
        return None
    
    result_vec = np.sum([glove_emb[a] for a in additions], axis=0)
    result_vec -= np.sum([glove_emb[s] for s in subtractions], axis=0)
    return result_vec

def best_match_glove(glove, vector, exclude_words):
    for word, _ in glove.similar_by_vector(vector, topn=10 + len(exclude_words)):
        if word not in exclude_words:
            return word
    return None

## BERT FUNCTIONS
def get_token_embedding_bert(tokenizer, embeddings, word):
    token_id = tokenizer.convert_tokens_to_ids(word.lower())
    if token_id == tokenizer.unk_token_id:
        return None
    
    return embeddings[token_id]

def embedding_calc_bert(tokenizer, embeddings, additions, subtractions):
    words = additions + subtractions
    missing = [word for word in words if get_token_embedding_bert(tokenizer, embeddings, word) is None]

    if missing:
        print(f"{', '.join(missing)} are not known to the Bert tokenizer")
        return None

    result_vec = np.sum([get_token_embedding_bert(tokenizer, embeddings, a) for a in additions], axis=0)
    result_vec -= np.sum([get_token_embedding_bert(tokenizer, embeddings, s) for s in subtractions], axis=0)

    return result_vec

def best_match_bert(tokenizer, embeddings, result_vec, exclude_words):
    similarities = cosine_similarity(result_vec.reshape(1, -1), embeddings)[0]
    sorted_ids = np.argsort(similarities)[::-1]

    for id in sorted_ids:
        token = tokenizer.convert_ids_to_tokens([id])[0]
        if token in exclude_words or token.isdigit() or token.startswith("[") or token.startswith("##"):
            continue

        return token
    return None
    
def main():
    print("Loading models...")
    glove = load_glove_model(GLOVE_PATH, CONVERTED_PATH)

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_embeddings = bert_model.embeddings.word_embeddings.weight.detach().numpy()

    user_input = ''

    print('---------------------')
    print('SIMPLE WORD CALCULATOR')
    print('type "exit" to quit')

    while True:
        print('\n---------------------')
        user_input = input('input calculation: ')
        if user_input == 'exit':
            break

        additions, subtractions = parse_input(user_input)
        input_words = additions + subtractions

        glove_result_vec = embedding_calc_glove(glove, additions, subtractions)
        if glove_result_vec is not None:
            print('glove = ', best_match_glove(glove, glove_result_vec, input_words))

        bert_result_vec = embedding_calc_bert(bert_tokenizer, bert_embeddings, additions, subtractions)
        if bert_result_vec is not None:
            print('bert  = ', best_match_bert(bert_tokenizer, bert_embeddings, bert_result_vec, input_words))

if __name__ == '__main__':
    main()