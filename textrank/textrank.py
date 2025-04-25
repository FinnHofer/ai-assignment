import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

text = """
Syria: searching for ways to end the violence 
The Syrian regime is increasingly isolated on the world stage. Interviewed on 7 January, Foreign Minister Guido Westerwelle called for Assad to resign and make way for a democratic fresh start in Syria. 
On 11 November, the Syrian opposition agreed to set up a new joint body, 
The deadlock in the United Nations Security Council continues. 
Foreign Minister Guido Westerwelle has repeatedly called upon China and Russia, both veto powers, to stop backing the Syrian regime. 
The Security Council has been struggling for months to agree on a united stance in relation to the Syrian regime. 
Bild: UN Secretary-General Ban Kimoon and Lakhdar Brahimi 
The conflict in Syria was also a dominant theme during the opening week of the United Nations General Assembly from 25 September to 1 October. 
Lakhdar Brahimi, the new Joint Special Representative of the Secretary-General and the Arab League, reported on developments to the Security Council. 
Brahimi took over from Kofi Annan when the latter chose not to extend his mandate. 
The plan has yet to be implemented. 
The UN observer mission UNSMIS, which was established to monitor implementation of the Annan plan, also came to an end on 19 August. 
Opposition platform gains international recognition 
Bild: Meeting of the Syrian opposition in Doha 
Syria’s opposition plays an important role in the planning for a peaceful future for the country. 
"""

def get_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def text_rank(sentences, similarities, min_sim=0):
    nx_graph = nx.Graph()

    for i, sentence in enumerate(sentences):
        nx_graph.add_node(i, label=sentence)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = similarities[i][j]
            if sim > min_sim:
                nx_graph.add_edge(i, j, weight=sim)
    
    scores = nx.pagerank(nx_graph, weight='weight')
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    for rank, (score, sentence) in enumerate(ranked_sentences[:3], 1):
        print(f"{rank}. ({score:.3f}) {sentence}")

    return nx_graph

def plot_graph(nx_graph, title):
    pos = nx.spring_layout(nx_graph, weight='weight', seed=42)
    labels = nx.get_node_attributes(nx_graph, 'label')
    edge_weights = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges]

    plt.figure(figsize=(10, 7))
    nx.draw(
        nx_graph, pos,
        with_labels=True,
        labels=labels,
        width=edge_weights,
        node_color='lightblue',
        font_size=7,
        edge_color='gray'
    )
    plt.gcf().canvas.manager.set_window_title(title)
    plt.show()

def main():
    sentences = get_sentences(text)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    tfidf_similarity = cosine_similarity(tfidf_matrix)

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    embedding_similarity = cosine_similarity(embeddings)

    print('TextRank - TF-IDF')
    nx_graph_tfidf = text_rank(sentences, tfidf_similarity)
    print('\n-----------------------------\n')

    print('TextRank - Embedding')
    nx_graph_embedding = text_rank(sentences, embedding_similarity)

    plot_graph(nx_graph_tfidf, 'TF-IDF - Graph')
    plot_graph(nx_graph_embedding, 'Embedding - Graph')


if __name__ == '__main__':
    main()