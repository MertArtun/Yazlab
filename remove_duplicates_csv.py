import pandas as pd


def remove_duplicates_from_csv(input_csv, output_csv):
    # CSV dosyasını yükle
    df = pd.read_csv(input_csv)

    # İlk olarak, tüm sütunlarda tam olarak aynı olan satırları kaldır
    initial_count = len(df)
    df_dedup = df.drop_duplicates()
    final_count = len(df_dedup)
    print(f"Toplam satır sayısı: {initial_count}")
    print(f"Tekrarlayan satırlar kaldırıldıktan sonra satır sayısı: {final_count}")

    # Eğer sadece 'sentence' sütununda tekrarı kaldırmak isterseniz:
    # df_dedup = df.drop_duplicates(subset=['sentence'])

    # Temizlenmiş veriyi yeni bir CSV dosyasına kaydet
    df_dedup.to_csv(output_csv, index=False)
    print(f"Temizlenmiş veri '{output_csv}' dosyasına kaydedildi.")


if __name__ == "__main__":
    input_csv = 'labeled_sentences.csv'  # Girdi CSV dosyanızın yolu
    output_csv = 'labeled_sentences_dedup.csv'  # Çıktı CSV dosyanızın yolu

    remove_duplicates_from_csv(input_csv, output_csv)
