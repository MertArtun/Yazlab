
import os
import certifi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# SSL sertifika yolunu ayarla
os.environ['SSL_CERT_FILE'] = certifi.where()

# NLTK için gerekli veri paketlerini indir
nltk.download('stopwords')
nltk.download('punkt')

# Stopword'leri ayarla
stop_words = set(stopwords.words('english'))

# Veri setini yükle
df = pd.read_csv('labeled_sentences_dedup.csv')

# Veri ön işleme
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df['processed_sentence'] = df['sentence'].apply(preprocess_text)

# Stil ayarları
sns.set(style="whitegrid")

# 1. Kategorilere Göre Cümle Sayısı Grafiği
def plot_sentence_counts(df):
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='category', data=category_counts, palette='viridis')
    plt.title('Kategorilere Göre Cümle Sayısı')
    plt.xlabel('Cümle Sayısı')
    plt.ylabel('Kategori')
    plt.tight_layout()
    plt.show()

# 2. Kelime Bulutları
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=stop_words,
                          collocations=False).generate(text)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.show()

def plot_wordclouds(df):
    categories = df['category'].unique()

    for category in categories:
        text = ' '.join(df[df['category'] == category]['processed_sentence'])
        title = f'Kelime Bulutu: {category}'
        generate_wordcloud(text, title)

# 3. En Sık Kullanılan Kelimelerin Grafikleri
def get_top_n_words(text, n=20):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    counter = Counter(tokens)
    return counter.most_common(n)

def plot_top_words(df, top_n=20):
    categories = df['category'].unique()

    for category in categories:
        text = ' '.join(df[df['category'] == category]['processed_sentence'])
        top_words = get_top_n_words(text, top_n)
        if not top_words:
            print(f"{category} kategorisi için yeterli kelime bulunamadı.")
            continue
        words, counts = zip(*top_words)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(counts), y=list(words), palette='magma')
        plt.title(f'En Sık Kullanılan {top_n} Kelime: {category}')
        plt.xlabel('Frekans')
        plt.ylabel('Kelime')
        plt.tight_layout()
        plt.show()

# 4. Genel Kelime Bulutu
def plot_general_wordcloud(df):
    all_text = ' '.join(df['processed_sentence'])
    generate_wordcloud(all_text, 'Genel Kelime Bulutu')

# 5. Kelime Frekansı Isı Haritası
def plot_word_frequency_heatmap(df):
    categories = df['category'].unique()
    all_text = ' '.join(df['processed_sentence'])
    tokens = all_text.split()
    counter = Counter(tokens)
    common_words = [word for word, freq in counter.most_common(100)]  # En yaygın 100 kelime

    # Her kategori için kelime frekansını hesaplama
    category_word_counts = {}

    for category in categories:
        text = ' '.join(df[df['category'] == category]['processed_sentence'])
        tokens = text.split()
        counter = Counter(tokens)
        category_word_counts[category] = [counter[word] for word in common_words]

    # DataFrame oluşturma
    word_freq_df = pd.DataFrame(category_word_counts, index=common_words)

    # Isı haritası
    plt.figure(figsize=(15, 20))
    sns.heatmap(word_freq_df, cmap='YlGnBu', linewidths=.5)
    plt.title('Kategorilere Göre Kelime Frekansı Isı Haritası')
    plt.xlabel('Kategori')
    plt.ylabel('Kelime')
    plt.show()

# 6. Duygu Analizi
def get_sentiment(sentence):
    blob = TextBlob(sentence)
    return blob.sentiment.polarity

def sentiment_analysis(df):
    df['sentiment'] = df['sentence'].apply(get_sentiment)

    # Kategorilere göre ortalama sentiment
    category_sentiment = df.groupby('category')['sentiment'].mean().reset_index()

    # Çubuk grafik
    plt.figure(figsize=(12, 8))
    sns.barplot(x='sentiment', y='category', data=category_sentiment, palette='coolwarm')
    plt.title('Kategorilere Göre Ortalama Duygu Skoru')
    plt.xlabel('Ortalama Duygu Skoru')
    plt.ylabel('Kategori')
    plt.tight_layout()
    plt.show()

# 7. Konu Modelleme (LDA)
def topic_modeling(df, num_topics=5):
    texts = df['processed_sentence'].apply(lambda x: x.split())

    # Dictionary ve Corpus oluşturma
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA modeli eğitme
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Topic'leri yazdırma
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx} \nWords: {topic}\n")

    # LDA görselleştirme
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("LDA görselleştirmesi 'lda_visualization.html' olarak kaydedildi.")

# 8. Sankey Diyagramı
def plot_sankey_diagram(df, top_n=5):
    categories = df['category'].unique()
    category_words = {}

    for category in categories:
        text = ' '.join(df[df['category'] == category]['processed_sentence'])
        top_words = get_top_n_words(text, top_n)
        category_words[category] = [word for word, freq in top_words]

    # Kaynak (source) ve hedef (target) listelerini oluşturma
    source = []
    target = []
    value = []
    labels = list(categories) + list(set([word for words in category_words.values() for word in words]))

    label_indices = {label: idx for idx, label in enumerate(labels)}

    for category, words in category_words.items():
        for word in words:
            source.append(label_indices[category])
            target.append(label_indices[word])
            value.append(1)  # Her ilişki için 1 değer veriyoruz

    # Sankey diyagramı oluşturma
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "blue"
        ),
        link = dict(
          source = source,
          target = target,
          value = value
      ))])

    fig.update_layout(title_text="Kategorilere Göre En Sık Kullanılan Kelimeler Sankey Diyagramı", font_size=10)
    fig.show()

# 9. Kelime Düzeyinde Embedding ve Görselleştirme
def word_embeddings_analysis(df):
    # Tüm metinlerden bir kelime listesi oluştur
    all_text = ' '.join(df['processed_sentence'])
    tokens = all_text.split()

    # Word2Vec modeli eğit
    model = gensim.models.Word2Vec([tokens], vector_size=100, window=5, min_count=2, workers=4)

    # Kelime vektörlerini al
    word_vectors = model.wv
    vocabs = list(word_vectors.index_to_key)
    word_vecs = word_vectors[vocabs]

    # Boyut indirgeme (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(word_vecs)

    # Görselleştirme
    plt.figure(figsize=(15, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', edgecolors='k')

    for i, word in enumerate(vocabs):
        plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
        if i >= 100:  # Çok fazla kelime için sınırlama
            break

    plt.title('Kelime Embedding PCA Görselleştirmesi')
    plt.xlabel('Bileşen 1')
    plt.ylabel('Bileşen 2')
    plt.show()

# 10. Cümle Düzeyinde Embedding ve Kümeleme
def sentence_embeddings_clustering(df):
    # Sentence Transformer modeli yükle
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Cümle embedding'lerini oluştur
    sentences = df['processed_sentence'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Boyut indirgeme (UMAP)
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)

    # Kümelere ayırma (K-Means)
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Görselleştirme
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'Küme {i}' for i in range(num_clusters)], title="Kümeler")
    plt.title('Cümle Embedding UMAP ve K-Means Kümeleme')
    plt.xlabel('Bileşen 1')
    plt.ylabel('Bileşen 2')
    plt.show()

# 11. Benzerlik Analizi ve Network Grafiği
def similarity_network(df):
    # İlk 100 cümleyi al (çok büyük veri setlerinde performans için)
    sample_df = df.head(100)
    sentences = sample_df['processed_sentence'].tolist()

    # Sentence Transformer modeli yükle
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Benzerlik matrisi
    similarity_matrix = cosine_similarity(embeddings)

    # NetworkX grafiği oluştur
    G = nx.Graph()

    # Düğümleri ekle
    for idx, sentence in enumerate(sentences):
        G.add_node(idx, sentence=sentence)

    # Kenarları ekle (benzerlik eşiğinin üzerinde olanlar)
    threshold = 0.7  # Benzerlik eşiği
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    # Grafiği çiz
    pos = nx.spring_layout(G, k=0.5)
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('Cümleler Arası Benzerlik Ağı')
    plt.axis('off')
    plt.show()

# 12. Heatmap (Kelime-Frekans Matrisi)
def plot_frequency_heatmap(df):
    vectorizer = CountVectorizer(max_features=50, stop_words='english')
    X = vectorizer.fit_transform(df['processed_sentence'])
    freq_matrix = X.toarray()
    words = vectorizer.get_feature_names_out()

    plt.figure(figsize=(15, 10))
    sns.heatmap(freq_matrix.T, cmap='YlGnBu', cbar=True, yticklabels=words)
    plt.title('Kelime-Frekans Isı Haritası')
    plt.xlabel('Cümleler')
    plt.ylabel('Kelimeler')
    plt.show()

# 13. Dendrogram (Hiyerarşik Kümeleme)
def plot_dendrogram(df):
    # Sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = df['processed_sentence'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Hiyerarşik kümeleme
    from scipy.cluster.hierarchy import linkage, dendrogram
    linked = linkage(embeddings, 'ward')

    plt.figure(figsize=(15, 10))
    dendrogram(linked,
               orientation='top',
               labels=None,
               distance_sort='descending',
               show_leaf_counts=False)
    plt.title('Cümlelerin Dendrogramı')
    plt.xlabel('Cümleler')
    plt.ylabel('Mesafe')
    plt.show()

# 14. Ana Fonksiyon
def main():
    print("1. Kategorilere Göre Cümle Sayısı Grafiği Oluşturuluyor...")
    plot_sentence_counts(df)

    print("2. Kelime Bulutları Oluşturuluyor...")
    plot_wordclouds(df)

    print("3. En Sık Kullanılan Kelimelerin Grafikleri Oluşturuluyor...")
    plot_top_words(df, top_n=20)

    print("4. Genel Kelime Bulutu Oluşturuluyor...")
    plot_general_wordcloud(df)

    print("5. Kelime Frekansı Isı Haritası Oluşturuluyor...")
    plot_word_frequency_heatmap(df)

    print("6. Duygu Analizi Yapılıyor...")
    sentiment_analysis(df)

    print("7. Konu Modelleme (LDA) Yapılıyor...")
    topic_modeling(df, num_topics=5)

    print("8. Sankey Diyagramı Oluşturuluyor...")
    plot_sankey_diagram(df, top_n=5)

    print("9. Kelime Düzeyinde Embedding ve Görselleştirme Yapılıyor...")
    word_embeddings_analysis(df)

    print("10. Cümle Düzeyinde Embedding ve Kümeleme Yapılıyor...")
    sentence_embeddings_clustering(df)

    print("11. Benzerlik Analizi ve Network Grafiği Oluşturuluyor...")
    similarity_network(df)

    print("12. Heatmap (Kelime-Frekans Matrisi) Oluşturuluyor...")
    plot_frequency_heatmap(df)

    print("13. Dendrogram (Hiyerarşik Kümeleme) Oluşturuluyor...")
    # Dendrogram büyük veri setlerinde uzun sürebilir, isterseniz yorum satırına alabilirsiniz.
    # plot_dendrogram(df.head(200))
    print("Tüm analizler tamamlandı.")

if __name__ == "__main__":
    main()
