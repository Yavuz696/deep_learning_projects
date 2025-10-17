🧠 MNIST Görüntü İşleme ve Yapay Sinir Ağı (ANN) Projesi

Bu proje, **MNIST el yazısı rakam veri seti** üzerinde temel **görüntü işleme** ve **yapay sinir ağı (ANN)** eğitimi gerçekleştirmektedir.  
Amaç, ön işleme (preprocessing) adımlarını kullanarak modelin rakamları daha iyi öğrenmesini sağlamaktır.

---

## 🚀 Kullanılan Teknolojiler

- **Python 3.x**
- **TensorFlow / Keras** → Model oluşturma ve eğitme
- **OpenCV (cv2)** → Görüntü işleme (blur, edge detection, histogram equalization)
- **NumPy** → Sayısal işlemler
- **Matplotlib** → Eğitim sürecinin görselleştirilmesi

---

## 🧩 Proje Adımları

### 1️⃣ Veri Seti Yükleme

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
MNIST veri seti, 28x28 boyutunda 70.000 el yazısı rakam görüntüsünden oluşur.

2️⃣ Görüntü Ön İşleme (Image Preprocessing)
Aşağıdaki işlemler her bir görüntüye uygulanır:

Histogram Eşitleme: Parlaklık farklarını dengeler.

Gaussian Blur: Gürültüyü azaltır.

Canny Edge Detection: Kenarları belirler.

Flattening: 28x28 → 784 boyutuna indirger.

Normalization: Piksel değerlerini 0–1 aralığına ölçekler.

3️⃣ Model Mimarisi (ANN)
python
Copy code
model = Sequential([
    Dense(128, activation="relu", input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
ReLU (Rectified Linear Unit): Gizli katmanlarda doğrusal olmayan öğrenme sağlar.

Dropout (0.2): Overfitting’i önlemek için nöronların %20’sini rastgele kapatır.

Softmax: Çıkışta her sınıf için olasılık üretir.

4️⃣ Modelin Derlenmesi
python
Copy code
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
5️⃣ Modelin Eğitimi
python
Copy code
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=2
)
Her epoch’ta model, tüm eğitim verisini bir kez görür.
Kayıp (loss) azalırken doğruluk (accuracy) artar.

6️⃣ Sonuçların Görselleştirilmesi
Eğitim süreci matplotlib ile görselleştirilmiştir:

Sol grafik: Eğitim ve doğrulama kaybı

Sağ grafik: Eğitim ve doğrulama doğruluğu

Overfitting belirtileri gözlemlenebilir; Dropout oranı veya epoch sayısı ile oynanarak iyileştirilebilir.

📊 Model Performansı
Metrik	Değer
Test Loss	~0.6
Test Accuracy	~0.90

(Not: Değerler eğitim sonucuna göre değişebilir.)

📁 Proje Yapısı
bash
Copy code
mnist_ann_project/
│
├── main.py              # Tüm proje kodu
├── README.md            # Bu dosya
└── requirements.txt     # Gerekli kütüphaneler (isteğe bağlı)
⚙️ Gereksinimler
bash
Copy code
pip install tensorflow opencv-python matplotlib numpy
📈 Geliştirme Önerileri
CNN (Convolutional Neural Network) ile performans artırılabilir.

Veri artırma (data augmentation) eklenerek modelin genelleme gücü artırılabilir.

Dropout oranı ve katman sayısı üzerinde deneyler yapılabilir.

🧑‍💻 Yazar
[Senin Adın]
Bilgisayar Mühendisliği Öğrencisi
📍 Türkiye
💡 “Veriyi anlamak, zekânın ilk adımıdır.”

yaml
Copy code

---

İstersen README’ye bir görsel (örneğin senin “Loss vs Accuracy” grafiğini) de ekleyebilirim.
Ekleyelim mi, hem sayfa daha profesyonel görünür?





```
