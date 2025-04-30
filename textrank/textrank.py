import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_PATH = PROJECT_ROOT / 'textrank' / 'text.txt'

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().replace('\n', ' ')

def get_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def text_rank(sentences, similarities, min_sim=0):
    nx_graph = nx.Graph()

    for i, sentence in enumerate(sentences):
        nx_graph.add_node(i, label=i)

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
    text = read_text_file(TEXT_PATH)
    sentences = get_sentences(text)

    tfidf_matrix = TfidfVectorizer().fit_transform(sentences)
    tfidf_similarity = cosine_similarity(tfidf_matrix)

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    embedding_similarity = cosine_similarity(embeddings)

    print('TextRank - TF-IDF')
    nx_graph_tfidf = text_rank(sentences, tfidf_similarity)
    print('\n-----------------------------\n')

    print('TextRank - Embedding')
    nx_graph_embedding = text_rank(sentences, embedding_similarity)

    #plot_graph(nx_graph_tfidf, 'TF-IDF - Graph')
    #plot_graph(nx_graph_embedding, 'Embedding - Graph')


if __name__ == '__main__':
    main()