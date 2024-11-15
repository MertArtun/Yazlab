YouTube Podcast Downloader ve Transkript İşleme Projesi

İçindekiler

	•	Proje Hakkında
	•	Özellikler
	•	Gereksinimler
	•	Kurulum
	•	1. Sanal Ortam Oluşturma ve Aktifleştirme
	•	2. Gerekli Paketlerin Kurulumu
	•	3. API Anahtarlarının Ayarlanması
	•	Kullanım
	•	4. YouTube Videolarının İndirilmesi
	•	5. Ses Dosyalarının Transkripte Dönüştürülmesi
	•	6. Transkriptlerin İşlenmesi ve Etiketlenmesi
	•	7. Veri Görselleştirme
	•	Dosya Yapısı
	•	Notlar
	•	Katkıda Bulunanlar
	•	Lisans

Proje Hakkında

Bu proje, YouTube’dan belirli kategorilerdeki (teknoloji, eğitim, spor, sanat, bilim) İngilizce podcast videolarını indirir, ses dosyalarını elde eder ve ardından bu ses dosyalarını metin transkriptlerine dönüştürür. Transkriptler temizlenir, cümlelere ayrılır, etiketlenir ve veri görselleştirme işlemleri için hazırlanır.

Özellikler

	•	YouTube Videolarının İndirilmesi: Belirli arama sorguları ve kategoriler kullanarak YouTube API üzerinden videoları arar ve indirir.
	•	Ses Dosyalarının Elde Edilmesi: İndirilen videoların ses dosyalarını MP3 formatında kaydeder.
	•	Transkript Oluşturma: Ses dosyalarını metin transkriptlerine dönüştürür.
	•	Veri İşleme: Transkriptleri temizler, cümlelere ayırır, lemmatize eder ve etiketler.
	•	Veri Görselleştirme: Etiketlenmiş verileri kullanarak kelime bulutları, kelime frekansları ve kategori dağılımları gibi görselleştirmeler oluşturur.

Gereksinimler

	•	Python 3.6 veya üzeri
	•	YouTube Data API v3 Anahtarı
	•	FFmpeg (YouTube videolarını ses dosyalarına dönüştürmek için)
	•	Sanal Ortam (Tavsiye edilir)

Kurulum

1. Sanal Ortam Oluşturma ve Aktifleştirme

Projenin bağımlılıklarını izole etmek için bir sanal ortam oluşturmanız tavsiye edilir.

# Proje dizininize gidin
cd /path/to/your/project

# Sanal ortam oluşturun (örneğin: venv)
python -m venv venv

# Sanal ortamı aktifleştirin
# macOS/Linux:
source venv/bin/activate

# Windows:
.\venv\Scripts\activate

2. Gerekli Paketlerin Kurulumu

Gerekli tüm Python paketlerini aşağıdaki komutla kurabilirsiniz:

pip install -r requirements.txt

Eğer requirements.txt dosyası yoksa, aşağıdaki komutu kullanabilirsiniz:

pip install yt_dlp requests isodate nltk pandas python-dotenv certifi matplotlib seaborn wordcloud textblob gensim pyLDAvis plotly scikit-learn sentence-transformers umap-learn networkx langdetect num2words

3. API Anahtarlarının Ayarlanması

YouTube Data API kullanabilmek için bir API anahtarına ihtiyacınız vardır.
	1.	Google Cloud Console üzerinden bir proje oluşturun ve YouTube Data API v3 için API anahtarı alın.
	2.	Proje dizininizde bir .env dosyası oluşturun ve aşağıdaki gibi doldurun:

YOUTUBE_API_KEY_1=YOUR_YOUTUBE_API_KEY_1
YOUTUBE_API_KEY_2=YOUR_YOUTUBE_API_KEY_2  # İkinci bir API anahtarınız varsa
AUDIO_FOLDER=audio_podcasts
MAX_RESULTS=200

	3.	config.py dosyasını oluşturun ve aşağıdaki kodu ekleyin:

# config.py

from dotenv import load_dotenv
import os

load_dotenv()

YOUTUBE_API_KEY_1 = os.getenv('YOUTUBE_API_KEY_1')
YOUTUBE_API_KEY_2 = os.getenv('YOUTUBE_API_KEY_2')
AUDIO_FOLDER = os.getenv('AUDIO_FOLDER', 'audio_podcasts')
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 200))

Kullanım

4. YouTube Videolarının İndirilmesi

downloader.py scripti, YouTube’dan videoları indirir ve ses dosyalarını kaydeder.

Terminalde aşağıdaki komutu çalıştırın:

python downloader.py

Bu script, belirlenen kategoriler için YouTube’da arama yapacak, videoları indirecek ve ses dosyalarını AUDIO_FOLDER içinde kaydedecektir.

Not: Script, API kotalarınızı hızlıca tüketebilir. API kullanımınızı takip edin ve gerekirse ek API anahtarları edinin.

5. Ses Dosyalarının Transkripte Dönüştürülmesi

Ses dosyalarını metin transkriptlerine dönüştürmek için bir ses tanıma (speech recognition) aracına ihtiyacınız vardır. Örneğin, OpenAI’nin whisper modelini kullanabilirsiniz.

Whisper Kurulumu

pip install git+https://github.com/openai/whisper.git

Not: Whisper’ı kullanmak için ffmpeg ve rust kurulu olmalıdır.
	•	FFmpeg Kurulumu:
	•	macOS (Homebrew ile):

brew install ffmpeg


	•	Windows:
	1.	FFmpeg İndir
	2.	İndirilen dosyayı açın ve bin klasöründeki ffmpeg.exe dosyasını sistem PATH’ine ekleyin.
	•	Linux (Debian/Ubuntu):

sudo apt update
sudo apt install ffmpeg


	•	Rust Kurulumu:

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh



Transkript Oluşturma

Ses dosyalarını transkripte dönüştürmek için:

whisper audio_podcasts/ --model large --language English --output_dir transcripts/

Bu komut, audio_podcasts/ klasöründeki ses dosyalarını transkripte dönüştürür ve sonuçları transcripts/ klasörüne kaydeder.

6. Transkriptlerin İşlenmesi ve Etiketlenmesi

Transkriptleri temizlemek, cümlelere ayırmak, lemmatize etmek ve etiketlemek için process_transcripts.py scriptini kullanın.

python process_transcripts.py

Bu script, transcripts/ klasöründeki metin dosyalarını işler ve sonuçları labeled_sentences.csv dosyasına kaydeder.

7. Veri Görselleştirme

Etiketlenmiş verileri görselleştirmek için visualize_data.py scriptini kullanabilirsiniz.

python visualize_data.py

Bu script, kelime bulutları, kelime frekansları ve kategori dağılımı gibi görselleştirmeler oluşturur.

Dosya Yapısı

Proje dizini aşağıdaki gibi yapılandırılmıştır:

project/
├── downloader.py            # YouTube videolarını indirmek için script
├── process_transcripts.py   # Transkriptleri işlemek ve etiketlemek için script
├── visualize_data.py        # Veriyi görselleştirmek için script
├── config.py                # Yapılandırma dosyası
├── requirements.txt         # Gerekli Python paketleri
├── .env                     # API anahtarları ve yapılandırma ayarları
├── audio_podcasts/          # İndirilen ses dosyaları
├── transcripts/             # Transkript dosyaları
├── labeled_sentences.csv    # İşlenmiş ve etiketlenmiş veriler
├── logs/                    # Log dosyaları
└── README.md                # Proje dokümantasyonu

Notlar

	•	FFmpeg Kurulumu: yt_dlp ve whisper kullanırken ses dosyalarını dönüştürmek için FFmpeg yüklü olmalıdır.
	•	SSL Sertifika Hatası: NLTK paketlerini indirirken SSL sertifika hatası alırsanız, Python’un sertifikalarını güncelleyin veya SSL doğrulamasını geçici olarak devre dışı bırakın:
  
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


	•	NLTK Sürüm Sorunu: punkt_tab not found hatası alırsanız, NLTK sürümünü 3.7’ye düşürün:

pip install nltk==3.7


	•	Sanal Ortamın Aktif Olduğundan Emin Olun: Tüm işlemleri yaparken sanal ortamınızın aktif olduğundan emin olun.

Katkıda Bulunanlar

  •	Halit Mert Artun
  •	Latif Atmaca



Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.
