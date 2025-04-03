import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

def most_similar_word(vector, input_words):
    """Find the most similar word that is not in the input words list."""

    similar_words = nlp.vocab.vectors.most_similar(vector.reshape(1, -1), n=10)
    similar_words = [nlp.vocab.strings[w].lower() for w in similar_words[0][0]]
    
    input_singulars = {nlp(word)[0].lemma_ for word in input_words}


    for output_word in similar_words:
        output_singular = nlp(output_word)[0].lemma_

        if output_word not in input_words and output_singular not in input_singulars:
            return output_word.title()
    
    return "No suitable word found"

def calc_vector(input_words, operations):
    """Compute the resulting vector based on the Input Words and Operations"""
    result_vector = None

    for i in range(0, len(input_words) - 1):
        word_vector = nlp(input_words[i]).vector

        if result_vector is None:
            result_vector = word_vector.copy()
        else:
            if operations[i] == '+':
                result_vector += word_vector
            elif operations[i] == '-':
                result_vector -= word_vector

    return result_vector


def parse_calc(calc):
    """Parse a given Calculation into the Words and Operations"""
    tokens = calc.lower().split()
    operation = None

    input_words = []
    operations = []
    
    for token in tokens:
        if token in {'+', '-'}:
            operation = token
        elif token in {'/', '*'}:
            print('"Only + and - operations are allowed"')
            return None, None
        else:
            if not (operation == None and len(operations) != 0):
                operations.append(operation)
                operation = None
            
            input_words.append(token)
    
    if len(input_words) == len(operations):
        return input_words, operations
    else:
        return None, None
    
def main():
    """Main function to handle user input and output results."""
    print('Enter a simple word calculation using + or -')
    print('+ or - must be separated by a space from words')
    print('Exit using CTRL + C')
    
    while True:
        print('\n--------------------------')
        user_input = input("Enter expression: ")
        
        input_words, operations = parse_calc(user_input)
        result_vector = calc_vector(input_words, operations)

        if result_vector is not None:
            result_word = most_similar_word(result_vector, input_words)
            print(f"= {result_word}")
        else:
            print("Invalid input. Please try again.")

if __name__ == '__main__':
    main()