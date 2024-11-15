
import os
import logging
from yt_dlp import YoutubeDL
from urllib.parse import quote
import requests
from config import (
    YOUTUBE_API_KEY_1,
    YOUTUBE_API_KEY_2,
    AUDIO_FOLDER,
    MAX_RESULTS
)
import isodate
from datetime import datetime, timedelta

# Loglama yapılandırması
logging.basicConfig(
    filename='logs/downloader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_downloaded_videos(file_path='downloaded_videos.txt'):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file.readlines())

def save_downloaded_video(video_id, file_path='downloaded_videos.txt'):
    with open(file_path, 'a') as file:
        file.write(f"{video_id}\n")

def fetch_videos(search_query, max_results=200, min_duration=3200, downloaded_videos=set()):
    """
    YouTube Data API kullanarak video araması yapar ve podcast videolarını filtreler.
    """
    print(f"Fetching videos for query: '{search_query}' with max results: {max_results}")
    search_query_encoded = quote(search_query)
    videos_with_podcast = []
    next_page_token = None

    while len(videos_with_podcast) < max_results:
        url = (
            f"https://www.googleapis.com/youtube/v3/search?"
            f"key={YOUTUBE_API_KEY_2}&q={search_query_encoded}&part=snippet&type=video&"
            f"order=date&maxResults=50&videoDuration=long&relevanceLanguage=en"
        )
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        print(f"API Request URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"API isteği başarısız oldu, hata kodu: {response.status_code}")
            print(f"API request failed with status code: {response.status_code}")
            print(f"API Response: {response.text}")
            break

        data = response.json()
        videos = data.get('items', [])
        if not videos:
            logging.warning("API cevabında 'items' anahtarı bulunamadı veya boş.")
            print("No videos found in the API response.")
            break

        # Podcast videolarını filtreleyin ve daha önce indirilenleri atlayın
        for video in videos:
            video_id = video['id']['videoId']
            if video_id in downloaded_videos:
                print(f"Skipping already downloaded video: {video_id}")
                logging.info(f"Zaten indirilen video atlanıyor: {video_id}")
                continue

            title = video['snippet']['title'].lower()
            description = video['snippet']['description'].lower()

            # Kategoriye bağlı olarak uygun anahtar kelimeleri belirleyin
            if 'technology' in search_query.lower() or 'tech' in search_query.lower():
                # Teknoloji anahtar kelimeleri
                tech_keywords = [
                    'technology', 'tech', 'ai', 'artificial intelligence', 'machine learning',
                    'data science', 'blockchain', 'cybersecurity', 'cloud computing',
                    'software development', 'startup', 'gadgets', 'internet of things',
                    'big data', 'programming', 'mobile technology', 'tech news',
                    'virtual reality', 'augmented reality', 'quantum computing',
                    'fintech', 'robotics', 'tech entrepreneurship', 'digital transformation',
                    'biotechnology', 'space technology', 'green technology', 'smart technology',
                    'e-commerce technology', 'gaming technology', 'healthtech', 'edtech',
                    'devops', 'ui ux', 'data analytics', 'edge computing'
                ]
                if any(keyword in title or keyword in description for keyword in tech_keywords):
                    videos_with_podcast.append(video)
            elif 'education' in search_query.lower():
                # Eğitim anahtar kelimeleri
                education_keywords = [
                    'education', 'educational', 'learning', 'academic', 'study', 'teaching',
                    'e-learning', 'online education', 'higher education', 'k-12 education',
                    'homeschooling', 'education technology', 'edtech', 'language learning',
                    'math education', 'science education', 'history education',
                    'literature education', 'education policy', 'special education',
                    'early childhood education', 'education leadership', 'curriculum development',
                    'education innovation', 'teacher training', 'educational psychology',
                    'student success', 'education research', 'education reform',
                    'adult education', 'distance learning', 'blended learning',
                    'educational theory', 'stem education', 'arts education',
                    'physical education', 'education administration', 'vocational education'
                ]
                if any(keyword in title or keyword in description for keyword in education_keywords):
                    videos_with_podcast.append(video)
            elif 'sports' in search_query.lower() or 'sport' in search_query.lower():
                # Spor anahtar kelimeleri
                sports_keywords = [
                    'sports', 'sport', 'sports talk', 'sports discussions', 'sports interviews',
                    'sports analysis', 'sports news', 'football', 'basketball', 'baseball',
                    'soccer', 'tennis', 'cricket', 'boxing', 'mixed martial arts', 'esports',
                    'sports management', 'sports psychology', 'sports coaching', 'athlete interviews',
                    'sports business', 'sports technology', 'sports training', 'sports history',
                    'sports strategy', 'sports performance'
                ]
                if any(keyword in title or keyword in description for keyword in sports_keywords):
                    videos_with_podcast.append(video)
            elif 'art' in search_query.lower() or 'arts' in search_query.lower():
                # Sanat anahtar kelimeleri
                art_keywords = [
                    'art', 'arts', 'art talk', 'art discussions', 'art interviews',
                    'art analysis', 'art history', 'fine arts', 'modern art',
                    'contemporary art', 'digital art', 'graphic design', 'painting',
                    'sculpture', 'photography', 'illustration', 'animation',
                    'visual arts', 'performance art', 'art criticism', 'art education',
                    'creative arts', 'art entrepreneurship', 'art and culture',
                    'street art', 'art therapy', 'museum', 'art exhibitions'
                ]
                if any(keyword in title or keyword in description for keyword in art_keywords):
                    videos_with_podcast.append(video)
            elif 'science' in search_query.lower() or 'scientific' in search_query.lower():
                # Bilim anahtar kelimeleri
                science_keywords = [
                    'science', 'scientific', 'science talk', 'science discussions', 'science interviews',
                    'science analysis', 'science news', 'biology', 'chemistry', 'physics',
                    'astronomy', 'geology', 'environmental science', 'medical science',
                    'space science', 'neuroscience', 'genetics', 'ecology', 'scientific research',
                    'science education', 'science technology', 'quantum physics', 'biotechnology',
                    'climate science', 'data science', 'scientific innovation', 'robotics science',
                    'science and society', 'forensic science', 'marine biology'
                ]
                if any(keyword in title or keyword in description for keyword in science_keywords):
                    videos_with_podcast.append(video)
            else:
                # Eğer kategoriye uymuyorsa atla
                continue

            if len(videos_with_podcast) >= max_results:
                break

        next_page_token = data.get('nextPageToken')
        if not next_page_token:
            break

    if not videos_with_podcast:
        print("No podcast videos found in the API response.")
        return []

    # Video ID'lerini alın
    video_ids = [video['id']['videoId'] for video in videos_with_podcast]
    video_ids_str = ','.join(video_ids)
    videos_url = (
        f"https://www.googleapis.com/youtube/v3/videos?"
        f"key={YOUTUBE_API_KEY_2}&id={video_ids_str}&part=contentDetails,snippet"
    )

    print(f"Fetching video details from URL: {videos_url}")
    videos_response = requests.get(videos_url)
    if videos_response.status_code != 200:
        logging.error(f"Video detaylarını alırken hata oluştu, hata kodu: {videos_response.status_code}")
        print(f"Failed to get video details with status code: {videos_response.status_code}")
        print(f"API Response: {videos_response.text}")
        return []

    videos_data = videos_response.json()
    video_items = videos_data.get('items', [])

    # Süreye ve dile göre filtreleme
    filtered_videos = []
    for idx, video in enumerate(video_items):
        video_id = videos_with_podcast[idx]['id']['videoId']

        # contentDetails alanının mevcut olduğunu kontrol edin
        content_details = video.get('contentDetails')
        if not content_details:
            print(f"Skipping video with missing contentDetails: {video_id}")
            logging.info(f"contentDetails eksik olan video atlanıyor: {video_id}")
            continue

        duration = content_details.get('duration')
        if not duration:
            print(f"Skipping video with missing duration: {video_id}")
            logging.info(f"duration bilgisi eksik olan video atlanıyor: {video_id}")
            continue

        duration_seconds = isodate.parse_duration(duration).total_seconds()

        snippet = video.get('snippet', {})
        default_language = snippet.get('defaultLanguage', 'en')
        default_audio_language = snippet.get('defaultAudioLanguage', 'en')

        if (default_language == 'en' or default_audio_language == 'en') and duration_seconds >= min_duration:
            videos_with_podcast[idx]['duration_seconds'] = duration_seconds
            filtered_videos.append(videos_with_podcast[idx])
        else:
            print(f"Skipping non-English or short video: {video_id} ({duration_seconds} seconds)")
            logging.info(f"İngilizce olmayan veya kısa video atlanıyor: {video_id} ({duration_seconds} saniye)")

    print(f"Found {len(filtered_videos)} podcast videos after filtering by duration and language.")
    return filtered_videos

def download_audio(video_url, output_folder):
    """
    YouTube videosunu indirir ve sesi MP3 formatında kaydeder.
    """
    try:
        print(f"Starting download for: {video_url}")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"Downloaded and converted to MP3: {video_url}")
        logging.info(f"{video_url} başarıyla indirildi ve MP3'e dönüştürüldü.")
    except Exception as e:
        logging.error(f"{video_url} indirirken hata oluştu: {e}")
        print(f"Error downloading {video_url}: {e}")

def main():
    search_categories = {
        "technology": (
            "technology podcast OR tech podcast OR tech talks OR technology discussions OR tech interviews OR "
            "AI podcast OR artificial intelligence podcast OR machine learning podcast OR data science podcast OR "
            "blockchain podcast OR cybersecurity podcast OR cloud computing podcast OR software development podcast OR "
            "startup podcast OR gadgets podcast OR internet of things podcast OR big data podcast OR programming podcast OR "
            "mobile technology podcast OR tech news podcast OR virtual reality podcast OR augmented reality podcast OR "
            "quantum computing podcast OR fintech podcast OR robotics podcast OR tech entrepreneurship podcast OR "
            "digital transformation podcast OR biotechnology podcast OR space technology podcast OR green technology podcast OR "
            "smart technology podcast OR e-commerce technology podcast OR gaming technology podcast OR healthtech podcast OR "
            "edtech podcast OR devops podcast OR ui ux podcast OR data analytics podcast OR edge computing podcast"
        ),
        "education": (
            "education podcast OR educational podcast OR learning podcast OR academic podcast OR study podcast OR teaching podcast OR "
            "education talks OR education discussions OR education interviews OR educational talks OR e-learning podcast OR "
            "online education podcast OR higher education podcast OR K-12 education podcast OR homeschooling podcast OR "
            "education technology podcast OR edtech podcast OR language learning podcast OR math education podcast OR "
            "science education podcast OR history education podcast OR literature education podcast OR education policy podcast OR "
            "special education podcast OR early childhood education podcast OR education leadership podcast OR curriculum development podcast OR "
            "education innovation podcast OR teacher training podcast OR educational psychology podcast OR student success podcast OR "
            "education research podcast OR education reform podcast OR adult education podcast OR distance learning podcast OR "
            "blended learning podcast OR educational theory podcast OR STEM education podcast OR arts education podcast OR physical education podcast OR "
            "education administration podcast OR vocational education podcast"
        ),
        "sports": (
            "sports podcast OR sport podcast OR sports talk podcast OR sports discussions podcast OR sports interviews podcast OR "
            "sports analysis podcast OR sports news podcast OR football podcast OR basketball podcast OR baseball podcast OR "
            "soccer podcast OR tennis podcast OR cricket podcast OR boxing podcast OR mixed martial arts podcast OR esports podcast OR "
            "sports management podcast OR sports psychology podcast OR sports coaching podcast OR athlete interviews podcast OR "
            "sports business podcast OR sports technology podcast OR sports training podcast OR sports history podcast OR sports strategy podcast OR "
            "sports performance podcast"
        ),
        "art": (
            "art podcast OR arts podcast OR art talk podcast OR art discussions podcast OR art interviews podcast OR "
            "art analysis podcast OR art history podcast OR fine arts podcast OR modern art podcast OR contemporary art podcast OR "
            "digital art podcast OR graphic design podcast OR painting podcast OR sculpture podcast OR photography podcast OR "
            "illustration podcast OR animation podcast OR visual arts podcast OR performance art podcast OR art criticism podcast OR "
            "art education podcast OR creative arts podcast OR art entrepreneurship podcast OR art and culture podcast OR street art podcast OR "
            "art therapy podcast OR museum podcast OR art exhibitions podcast"
        ),
        "science": (
            "science podcast OR scientific podcast OR science talk podcast OR science discussions podcast OR science interviews podcast OR "
            "science analysis podcast OR science news podcast OR biology podcast OR chemistry podcast OR physics podcast OR astronomy podcast OR "
            "geology podcast OR environmental science podcast OR medical science podcast OR space science podcast OR neuroscience podcast OR "
            "genetics podcast OR ecology podcast OR scientific research podcast OR science education podcast OR science technology podcast OR "
            "quantum physics podcast OR biotechnology podcast OR climate science podcast OR data science podcast OR scientific innovation podcast OR "
            "robotics science podcast OR science and society podcast OR forensic science podcast OR marine biology podcast"
        )
    }
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

    # Daha önce indirilen videoları yükleyin
    downloaded_videos = load_downloaded_videos()

    for category, search_query in search_categories.items():
        print(f"Starting downloader with search query: '{search_query}'")
        logging.info(f"Başlangıç: '{search_query}' araması yapılıyor.")

        # Kategori klasörünü oluşturun
        category_folder = os.path.join(AUDIO_FOLDER, category)
        os.makedirs(category_folder, exist_ok=True)

        videos = fetch_videos(search_query, max_results=MAX_RESULTS, min_duration=3200, downloaded_videos=downloaded_videos)
        if not videos:
            logging.info(f"İndirilecek video bulunamadı: {category}")
            print(f"No videos to download for query: {search_query}")
            continue

        for item in videos:
            video_id = item['id'].get('videoId')
            if not video_id:
                print("No video ID found, skipping...")
                continue
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"Processing video: {video_url}")
            logging.info(f"İndiriliyor: {item['snippet']['title']}")
            download_audio(video_url, category_folder)
            # İndirilen videoyu kaydet
            save_downloaded_video(video_id)

    print("Downloader script completed.")

if __name__ == "__main__":
    main()