import os
import re
import nltk
import pandas as pd

# NLTK paketlerini indir (ilk çalıştırmada gereklidir)
nltk.download('punkt')

from nltk.tokenize import sent_tokenize


def clean_text(text):
    """
    Metni temizler:
    - Özel karakterleri ve noktalama işaretlerini kaldırır.
    - Fazla boşlukları tek boşluk yapar.
    - Metni küçük harfe dönüştürür.
    """
    # Özel karakterleri ve noktalama işaretlerini kaldırma
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Fazla boşlukları tek boşluk yapma
    text = re.sub(r'\s+', ' ', text)

    # Baş ve sondaki boşlukları kaldırma
    text = text.strip()

    # Metni küçük harfe dönüştürme
    text = text.lower()

    return text


def split_into_sentences(text):
    """
    Metni cümlelere ayırır.
    """
    return sent_tokenize(text)


def process_transcripts(transcript_dir):
    """
    Transkriptleri işler, temizler, cümlelere ayırır ve etiketler.
    """
    all_data = []

    # Kategori klasörlerini dolaş
    for category in os.listdir(transcript_dir):
        category_path = os.path.join(transcript_dir, category)
        if os.path.isdir(category_path):
            # Kategori içindeki dosyaları dolaş
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    # Metni temizle
                    cleaned_text = clean_text(text)

                    # Metni cümlelere ayır
                    sentences = split_into_sentences(cleaned_text)

                    # Her cümleyi etiketle
                    for sentence in sentences:
                        if sentence.strip():
                            all_data.append({
                                'sentence': sentence.strip(),
                                'category': category
                            })

                    print(f"Processed {file_path}")
    return all_data


def save_data(data, output_csv):
    """
    Veriyi CSV dosyasına kaydeder.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")


def main():
    transcript_dir = 'transcripts'  # Transkriptlerin bulunduğu klasör
    output_csv = 'labeled_sentences.csv'  # Çıktı CSV dosyası

    data = process_transcripts(transcript_dir)
    save_data(data, output_csv)


if __name__ == '__main__':
    main()
