import os
import re
import nltk
import pandas as pd
from langdetect import detect
from num2words import num2words

# SSL sertifikası hatasını gidermek için (gerekliyse)
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Uyarıları gizlemek için
import warnings
warnings.filterwarnings("ignore")

# NLTK paketlerini indir (ilk çalıştırmada gereklidir)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Kısaltmalar sözlüğü
contractions_dict = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not"

}

def expand_contractions(text):
    contractions_pattern = re.compile(r'\b({})\b'.format('|'.join(re.escape(k) for k in contractions_dict.keys())),
                                      flags=re.IGNORECASE)
    def replace(match):
        matched = match.group(0)
        lower_matched = matched.lower()
        expanded = contractions_dict.get(lower_matched, matched)
        return expanded
    expanded_text = contractions_pattern.sub(replace, text)
    return expanded_text

def remove_filler_words(text):
    filler_words = [
        'umm', 'uh', 'ah', 'like', 'you know', 'so', 'actually',
        'basically', 'right', 'well', 'um', 'er', 'eh', 'hmm', 'huh',
        'oh', 'ah', 'just', 'kind of', 'sort of', 'literally', 'really',
        'totally', 'seriously', 'okay', 'ok', 'you see', 'i mean',
        'you know what i mean', 'know what i mean'
    ]
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in filler_words) + r')\b', flags=re.IGNORECASE)
    return pattern.sub('', text)

def convert_numbers(text):
    def replace_number(match):
        number = match.group()
        try:
            number_int = int(number)
            word = num2words(number_int).replace('-', ' ')
            return word
        except ValueError:
            return number
    return re.sub(r'\b\d+\b', replace_number, text)

def clean_text(text):
    text = expand_contractions(text)
    text = remove_filler_words(text)
    text = convert_numbers(text)
    # Noktalama işaretlerini koruyoruz
    text = re.sub(r'[^\w\s\.\?!]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

def process_transcripts(transcript_dir, output_dir):
    all_data = []

    # Kategori klasörlerini dolaş
    for category in os.listdir(transcript_dir):
        category_path = os.path.join(transcript_dir, category)
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")
            cleaned_sentences = []
            # Kategori içindeki dosyaları dolaş
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    print(f"Original text from {filename}:\n{text[:200]}...\n")  # İlk 200 karakteri yazdır

                    # Metni temizle
                    cleaned_text = clean_text(text)
                    print(f"Cleaned text for {filename}:\n{cleaned_text[:200]}...\n")

                    # Metni lemmatize et
                    lemmatized_text = lemmatize_text(cleaned_text)
                    print(f"Lemmatized text for {filename}:\n{lemmatized_text[:200]}...\n")

                    # Metni cümlelere ayır
                    sentences = sent_tokenize(lemmatized_text)
                    print(f"Sentences from {filename}: {sentences}\n")

                    # Her cümleyi kontrol et
                    for sentence in sentences:
                        sentence = sentence.strip()
                        word_count = len(sentence.split())
                        if 3 <= word_count <= 50 and is_english(sentence):
                            cleaned_sentences.append(sentence)
                            # Tüm verileri 'all_data' listesine ekle
                            all_data.append({
                                'sentence': sentence,
                                'category': category
                            })

                    print(f"Processed {file_path}\n")

            # Temizlenmiş cümleleri kategoriye özel bir dosyaya kaydet
            if cleaned_sentences:
                output_category_dir = os.path.join(output_dir, f"cleaned_{category}")
                os.makedirs(output_category_dir, exist_ok=True)
                output_file_path = os.path.join(output_category_dir, f"{category}_cleaned.txt")
                with open(output_file_path, 'w', encoding='utf-8') as f_out:
                    for sentence in cleaned_sentences:
                        f_out.write(sentence + '\n')
                print(f"Saved cleaned sentences to {output_file_path}\n")
            else:
                print(f"No sentences collected for category '{category}'.\n")

    # Tüm verileri içeren bir CSV dosyası kaydet
    if all_data:
        df = pd.DataFrame(all_data)
        output_csv = os.path.join(output_dir, 'labeled_sentences.csv')
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
        print(f"Total sentences collected: {len(all_data)}")
    else:
        print("No data collected.")

def main():
    transcript_dir = 'transcripts'  # Transkriptlerin bulunduğu klasör
    output_dir = '.'  # Temizlenmiş verilerin kaydedileceği ana klasör (mevcut klasör)

    process_transcripts(transcript_dir, output_dir)

if __name__ == '__main__':
    main()