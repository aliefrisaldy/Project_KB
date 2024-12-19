import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Sesuaikan Path Ke Dataset
data = pd.read_csv(r'D:\The Journey\Semester 3\Kecerdasan Buatan\Prediksi Cuaca\Dataset\cuaca.csv')

print("Data Awal:")
print(data.head())
print(data.columns)
data.columns = data.columns.str.strip()

X = data[['Temperature', 'Humidity', 'WindSpeed', 'Pressure']]  
y = data['Weather Condition']   

scaler = RobustScaler()
X = scaler.fit_transform(X)

print("\nFitur setelah normalisasi dengan RobustScaler:")
print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nJumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))

k = 5
model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) 
print(f"Akurasi: {accuracy:.2f}%") 
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

k_values = range(1, 21)  
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred) * 100)  # Simpan akurasi dalam persen

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
plt.xlabel('Jumlah Tetangga (K)')
plt.ylabel('Akurasi (%)')
plt.title('Optimasi Nilai K pada K-NN')
plt.xticks(k_values)
plt.grid()
plt.show()

#Masukan Data Baru Yang Ingin Diuji Dan Diprediksi
data_baru = [[29.79, 30.65, 14.44, 1010.02]]  
data_baru = pd.DataFrame(data_baru, columns=['Temperature', 'Humidity', 'WindSpeed', 'Pressure'])
data_baru = scaler.transform(data_baru)
prediksi = model.predict(data_baru)
print("\nPrediksi untuk data baru:", prediksi)
