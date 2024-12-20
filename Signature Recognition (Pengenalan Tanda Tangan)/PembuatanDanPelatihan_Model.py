import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


DATASET_PATH = r"D:\The Journey\Semester 3\Kecerdasan Buatan\dataset"  # Sesuaikan Path ke folder dataset utama (Data Mentah)
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    

    if os.path.isdir(person_path) and person not in ["train", "test"]:
        images = os.listdir(person_path)
        

        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        
        # Buat folder teman di dalam train dan test
        os.makedirs(os.path.join(TRAIN_PATH, person), exist_ok=True)
        os.makedirs(os.path.join(TEST_PATH, person), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(TRAIN_PATH, person, img))
        

        for img in test_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(TEST_PATH, person, img))

print("Gambar berhasil dipisahkan ke dalam train dan test folder!")


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'  
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Output sesuai jumlah kelas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_generator, epochs=10, validation_data=test_generator)


import matplotlib.pyplot as plt

# Plot akurasi
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

best_train_acc = max(history.history['accuracy']) * 100
best_val_acc = max(history.history['val_accuracy']) * 100

print(f"Akurasi training: {best_train_acc:.2f}%")
print(f"Akurasi validation: {best_val_acc:.2f}%")

model.save("Model_Signature_Recognition_UAS.h5")
print("Model berhasil disimpan sebagai 'Model_Signature_Recognition_UAS.h5'!")
