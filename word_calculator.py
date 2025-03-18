import spacy

nlp = spacy.load('en_core_web_lg')

def most_similar_word(word_vector, input_words):
    similar_words = nlp.vocab.vectors.most_similar(word_vector.reshape(1, -1), n=10)

    words = [nlp.vocab.strings[w].lower() for w in similar_words[0][0]]

    for word in words:
        lemma = nlp(word)[0].lemma_

        input_lemmas = [nlp(input_word)[0].lemma_ for input_word in input_words]

        if (word not in input_words and lemma not in input_lemmas and word not in input_lemmas):
            return word
    
    return "no word found"

def calc_result_vector(calc):
    calc = calc.lower().split()

    combined_words = []
    result_vector = None
    operation = None
    for item in calc:
        if item in ['+', '-']:
            operation = item
        else:
            combined_words.append(item.lower())

            if (result_vector is None):
                result_vector = nlp(item).vector
            elif (operation == '+'):
                result_vector += nlp(item).vector
            elif (operation == '-'):
                result_vector -= nlp(item).vector

    return result_vector, combined_words

if __name__ == '__main__':
    print('Enter simple word calculation using + or -')
    print('exit using CTRL + C')
    while True:
        print('--------------------------\n')
        calc = input()
        result_vector, input_words  = calc_result_vector(calc)
        result_word = most_similar_word(result_vector, input_words)

        print("= " + result_word)
