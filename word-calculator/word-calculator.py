import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

def most_similar_word(vector, input_words):
    """Find the most similar word that is not in the input words list."""
    similar_words = nlp.vocab.vectors.most_similar(vector.reshape(1, -1), n=10)
    words = [nlp.vocab.strings[w].lower() for w in similar_words[0][0]]
    
    input_lemmas = {nlp(word)[0].lemma_ for word in input_words}
    
    for word in words:
        lemma = nlp(word)[0].lemma_
        if word not in input_words and lemma not in input_lemmas:
            return word
    
    return "No suitable word found"

def calculate_vector(expression):
    """Compute the resulting vector based on a word arithmetic expression."""
    tokens = expression.lower().split()
    result_vector = None
    operation = None
    input_words = []
    
    for token in tokens:
        if token in {'+', '-'}:
            operation = token
        else:
            input_words.append(token)
            word_vector = nlp(token).vector
            
            if result_vector is None:
                result_vector = word_vector.copy()
            elif operation == '+':
                result_vector += word_vector
            elif operation == '-':
                result_vector -= word_vector
    
    return result_vector, input_words

def main():
    """Main function to handle user input and output results."""
    print('Enter a simple word calculation using + or -')
    print('+ or - must be separated by a space from words')
    print('Exit using CTRL + C')
    
    while True:
        print('\n--------------------------')
        user_input = input("Enter expression: ")
        
        result_vector, input_words = calculate_vector(user_input)
        if result_vector is not None:
            result_word = most_similar_word(result_vector, input_words)
            print(f"= {result_word}")
        else:
            print("Invalid input. Please try again.")

if __name__ == '__main__':
    main()