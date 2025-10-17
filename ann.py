"""
tensorflow: keras ile modeli oluşturma ve eğitme işlemlerini yapar
matlotlib: eğitim sürecini görselleştirmek için kullanılır
cv2:opencv görüntü işleme kütüphanesi, görüntüleri yükleme ve işleme işlemlerini yapar
numpy: sayısal işlemler ve dizilerle çalışmak için kullanılır ama zaten bu tensflow içinde var






"""

import cv2 #opencv
import numpy as np #sayısal işlemler için
import matplotlib.pyplot as plt# görselleştirme için

from tensorflow.keras.datasets import mnist #mnist veri seti
from tensorflow.keras.models import Sequential #ANN modeli için
from tensorflow.keras.layers import Dense,Dropout #ANN katmanları
from tensorflow.keras.optimizers import Adam #adaptif momentum ,optimizer 

# MNIST veri setini yükle
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

#image preprocessing
img=x_train[5] # ilk resmi al

stages ={"original": img}

# histogram eşitleme 
eq=cv2.equalizeHist(img) # histogram eşitleme
stages["histogram equalized"]=eq

#gaussian blur grültü azaltma
blur=cv2.GaussianBlur(eq,(5,5),0)
stages["gaussian blur"]=blur

#kenar canny ile
edges=cv2.Canny(blur,50,150) # alt ve üst eşik değeri bu değerin üzerinde veya altındaysa  
stages["canny kenarlari"]=edges

#görselleştirme
fig,axes = plt.subplots(2,2,figsize=(6,6))
axes=axes.flat
for ax,(title,im) in zip(axes,stages.items()):
    ax.imshow(im,cmap="gray")
    ax.set_title(title)
    ax.axis("off")


plt.suptitle("görüntü işleme adımları")
plt.tight_layout()
plt.show()


# preprocessing tüm veri setine uygulamak
def preprocess_image(img):
    #histogram eşitleme 

    #guassian blur

    #canny kenar bulma

    #flattening 28x28 --> 784

    #normalization 0-255 --> 0-1 ölçekleme
    img_eq=cv2.equalizeHist(img)#çok parlak veya çok kısık olan kısımları dengeler

    img_blur=cv2.GaussianBlur(img_eq,(5,5),0)# bir tür bulanıklaştırma işlemidir,gürültüyü azaltır

    img_edges=cv2.Canny(img_blur,50,150)# köşeleri buluruz

    features= img_edges.flatten()/255.0# çok boyutlu olan diziyi tek boyuta indirger

    return features

num_train=10000
num_test=2000

x_train= np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub=y_train[:num_train]

x_test= np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub=y_test[:num_test]

# ann modelini tanımlamak
model=Sequential([
   Dense(128,activation="relu",input_shape=(784,)),
   Dropout(0.5),
   Dense(64,activation="relu"),
   Dense(10,activation="softmax"),

]
)

# modeli derlemek compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),#öğrenme hızı
    loss="sparse_categorical_crossentropy",# 
    metrics=["accuracy"]

)
print(model.summary())#nasıl bir model yaptığımızı gösterir

#ann modelini eğitmek ann model training
history=model.fit(
    x_train,y_train_sub,
    validation_data=(x_test,y_test_sub),
    epochs=50,
    batch_size=32,
    verbose=2
)

#modelin performansını değerlendirme aşaması
test_loss,test_acc=model.evaluate(x_test,y_test_sub)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
#plot training history 

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
