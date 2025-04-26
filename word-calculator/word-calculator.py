import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import re
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

glove_path = "glove.6B.50d.txt"
converted_path = "glove.6B.50d.word2vec.txt"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
embedding_matrix = model.embeddings.word_embeddings.weight.detach().numpy()

def load_glove_model(glove_path_txt, converted_path):
    if not os.path.exists(converted_path):
        glove2word2vec(glove_path_txt, converted_path)
    model = KeyedVectors.load_word2vec_format(converted_path, binary=False)
    return model

def format_input(usr_input):
    calc = usr_input.replace(" ", "")
    if not calc.startswith(('+', '-')):
        calc = '+' + calc

    # Split into list of tuples: [('+', 'king'), ('-', 'man'), ('+', 'woman')]
    tokens = re.findall(r'([+-])([^\+-]+)', calc)

    additions = [word for sign, word in tokens if sign == '+']
    subtractions = [word for sign, word in tokens if sign == '-']
    all_words = additions + subtractions

    return all_words, additions, subtractions

## GLOVE FUNCTIONS
def embedding_calc_glove(glove, input_words, additions, subtractions):
    glove_emb = {word: glove[word] for word in input_words if word in glove}
    missing = [word for word in input_words if word not in glove]

    if missing:
        print(f"{', '.join(missing)} are not in the GloVe model")
        return None
    
    result_vec = np.sum([glove_emb[a] for a in additions], axis=0)
    result_vec -= np.sum([glove_emb[s] for s in subtractions], axis=0)
    return result_vec

def get_best_match_glove(glove, result_vec, input_words):
    glove_most_similar = glove.similar_by_vector(result_vec, topn=10 + len(input_words))
    return [(word, sim) for word, sim in glove_most_similar if word not in input_words][0][0]

## BERT FUNCTIONS
def get_word_embedding_bert(word):
    token_id = tokenizer.convert_tokens_to_ids(word.lower())
    if token_id == tokenizer.unk_token_id:
        return None
    
    return embedding_matrix[token_id]

def embedding_calc_bert(input_words, additions, subtractions):
    missing = [word for word in input_words if get_word_embedding_bert(word) is None]

    if missing:
        print(f"{', '.join(missing)} are not known to the tokenizer")
        return None


    result_vec = np.sum([get_word_embedding_bert(a) for a in additions], axis=0)
    result_vec -= np.sum([get_word_embedding_bert(s) for s in subtractions], axis=0)

    return result_vec

def get_best_match_bert(result_vec, input_words):
    similarities = cosine_similarity(result_vec.reshape(1, -1), embedding_matrix)[0]
    desc_similar_tokens = tokenizer.convert_ids_to_tokens(np.argsort(similarities)[::-1])

    most_similar_token = ''
    for token in desc_similar_tokens:
        if token in input_words or token.isdigit() or token.startswith("[") or token.startswith("##"):
            continue
        else:
            most_similar_token = token
            break

    return most_similar_token
    
def main():
    glove = load_glove_model(glove_path, converted_path)
    user_input = ''

    print('---------------------')
    print('SIMPLE WORD CALCULATOR')
    print('type "exit" to quit')

    while user_input != 'exit':
        print('\n---------------------')
        user_input = input('input calculation: ')
        if user_input == '' or user_input == 'exit':
            continue

        input_words, additions, subtractions = format_input(user_input)

        glove_result_vec = embedding_calc_glove(glove, input_words, additions, subtractions)
        if glove_result_vec is not None:
            print('glove= ', get_best_match_glove(glove, glove_result_vec, input_words))

        bert_result_vec = embedding_calc_bert(input_words, additions, subtractions)
        if bert_result_vec is not None:
            print('bert= ', get_best_match_bert(bert_result_vec, input_words))

if __name__ == '__main__':
    main()