import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import re

def load_glove_model(glove_path_txt, converted_path):
    if not os.path.exists(converted_path):
        glove2word2vec(glove_path_txt, converted_path)
    model = KeyedVectors.load_word2vec_format(converted_path, binary=False)
    return model

def format_input(calc):
    calc = calc.replace(" ", "")
    if not calc.startswith(('+', '-')):
        calc = '+' + calc

    # Split into list of tuples: [('+', 'king'), ('-', 'man'), ('+', 'woman')]
    tokens = re.findall(r'([+-])([^\+-]+)', calc)

    additions = [word for sign, word in tokens if sign == '+']
    subtractions = [word for sign, word in tokens if sign == '-']
    all_words = additions + subtractions

    return all_words, additions, subtractions

def embedding_math(embeddings, additions, subtractions):
    try:
        result_vec = np.sum([embeddings[a] for a in additions], axis=0) - np.sum([embeddings[s] for s in subtractions], axis=0)
        return result_vec
    except KeyError:
        print(f"Word not found in embeddings.")
        return None

def main():
    glove_path = "glove.6B.50d.txt"
    converted_path = "glove.6B.50d.word2vec.txt"

    glove = load_glove_model(glove_path, converted_path)

    user_input = ''

    print('---------------------')
    print('SIMPLE WORD CALCULATOR')
    print('type "exit" to quit')

    while user_input != 'exit':
        print('\n---------------------')
        user_input = input('input calculation: ')
        input_words, additions, subtractions = format_input(user_input)

        glove_emb = {word: glove[word] for word in input_words if word in glove}
        missing = [word for word in input_words if word not in glove]
        if missing:
            print(f"! These words are not in the GloVe model: {', '.join(missing)} !")
            continue

        result_vec = embedding_math(glove_emb, additions, subtractions)

        most_similar = glove.similar_by_vector(result_vec, topn=10 + len(input_words))
        best_match = [(word, sim) for word, sim in most_similar if word not in input_words][0][0]
        print(best_match)

if __name__ == '__main__':
    main()