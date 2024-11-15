import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasındaki çevresel değişkenleri yükle
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# YouTube Data API anahtarı
YOUTUBE_API_KEY_1 = os.getenv('YOUTUBE_API_KEY_1')
YOUTUBE_API_KEY_2 = os.getenv('YOUTUBE_API_KEY_2')

# Maksimum sonuç sayısı
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 100))

# Ses dosyalarının kaydedileceği ana klasör
AUDIO_FOLDER = os.getenv('AUDIO_FOLDER', "/Users/halitartun/Desktop/yazlabses")

# Log dosyası
LOG_FILE = os.path.join('logs', 'downloader.log')
