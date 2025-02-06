import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import itertools

# Veri kümesi ve ses dosyalarının bulunduğu klasör
data_csv = r'C:\Users\tunak\BU BILGISAYRDAKI KODLAR\personal\projeler\Sound classificaiton\SONAR\DATA\balanced_data.csv'
audio_dir = r'C:\Users\tunak\BU BILGISAYRDAKI KODLAR\personal\projeler\Sound classificaiton\SONAR\DATA\shortened_dataset'  # Ses dosyalarının bulunduğu dizin
output_dir = r'C:\Users\tunak\BU BILGISAYRDAKI KODLAR\personal\projeler\Sound classificaiton\SONAR\DATA\OUTPUT2'  # Görsellerin kaydedileceği dizin

# Çıkış dizinini oluştur (varsa hata vermez)
os.makedirs(output_dir, exist_ok=True)

# Veriyi yükleme
data = pd.read_csv(data_csv)

# Özellik çıkarma (MFCC'ler) ve etiketleme
X = []
y = []
classes_seen = set()
sample_waveforms = {}
sample_mfccs = {}

for index, row in data.iterrows():
    file_name = row['slice_file_name']  # Ses dosyasının adı (CSV'deki doğru sütun adı)
    label = row['class']  # Etiket (CSV'deki doğru sütun adı)

    # Ses dosyasının tam yolu
    file_path = os.path.join(audio_dir, file_name)

    try:
        # Dosya yolu geçerli mi kontrol et
        if not os.path.exists(file_path):
            print(f"Hata: {file_path} dosyası mevcut değil.")
            continue

        # Ses dosyasını yükle
        y_signal, sr = librosa.load(file_path, sr=None)

        # MFCC çıkarma
        mfcc = librosa.feature.mfcc(y=y_signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        X.append(mfcc_mean)
        y.append(label)

        # Görselleştirme için her sınıftan bir örnek kaydet
        if label not in classes_seen:
            classes_seen.add(label)
            sample_waveforms[label] = (y_signal, sr)
            sample_mfccs[label] = mfcc

    except Exception as e:
        print(f"Hata: {file_path} dosyası işlenemedi. {e}")

# Özellikleri ve etiketleri numpy dizisine çevir
X = np.array(X)
y = np.array(y)

# Eğer X veya y boşsa, hata mesajı ver ve işlemi sonlandır
if X.shape[0] == 0 or y.shape[0] == 0:
    print("Veri kümesi boş, model eğitilemez.")
else:
    # Veri kümesini eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalizasyon
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Lojistik regresyon modeli oluşturma ve eğitme
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Test seti üzerinde değerlendirme
    y_pred = model.predict(X_test)

    print("Doğruluk:", model.score(X_test, y_test))
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

    # Karışıklık Matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Karışıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(len(set(y)))
    plt.xticks(tick_marks, set(y), rotation=45)
    plt.yticks(tick_marks, set(y))

    # Hücrelere sayıları ekleyelim
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Sınıflandırma Raporunu Bar Grafikle Görselleştirme
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Bar grafiği ile metriklerin görselleştirilmesi
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Sınıflandırma Raporu')
    plt.ylabel('Metrik Değerleri')
    plt.xlabel('Sınıf')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'))
    plt.close()

    # Dalga formu ve MFCC görselleştirme
    for label, (waveform, sr) in sample_waveforms.items():
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(waveform)/sr, len(waveform)), waveform)
        plt.title(f"Dalga Formu - {label}")
        plt.xlabel("Zaman (s)")
        plt.ylabel("Genlik")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"waveform_{label}.png"))
        plt.close()

    for label, mfcc in sample_mfccs.items():
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm')
        plt.colorbar()
        plt.title(f"MFCC - {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mfcc_{label}.png"))
        plt.close()
