import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model('Model_Signature_Recognition_UAS.h5')

model.summary()

# Daftar nama pemilik Tanda Tangan sesuai label yang digunakan saat pelatihan
class_names = ['Agil', 'Alif', 'Amirul', 'Ari', 'Aulia', 'Fadhlur', 'Fitra', 'Hajera', 'Hajir', 'Ihzan', 'Jesica', 'Juan', 'Mujahid', 'Nayla', 'Nur', 'Rasya', 'Rut', 'Siti', 'Teguh', 'Yunus']  # Sesuaikan dengan nama pemilik


def prepare(filepath):
    img = image.load_img(filepath, target_size=(126, 126))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

filepath = r'D:\The Journey\Semester 3\Kecerdasan Buatan\Kode Program\s.jpg'  # Ganti dengan path gambar sebagai data baru yang ingin di prediksi
img_array = prepare(filepath)

predictions = model.predict(img_array)


predicted_class_index = np.argmax(predictions[0])  
predicted_class = class_names[predicted_class_index]  


print(f"Gambar diprediksi sebagai Tanda Tangan: {predicted_class}")

