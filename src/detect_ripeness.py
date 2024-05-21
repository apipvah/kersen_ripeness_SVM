import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread_collection
from sklearn.model_selection import train_test_split
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ---------------------------- PRE-PROCESSING ----------------------------
# digunakan untuk memproses gambar sebelum dilakukan klasifikasi
def preprocess(img):
    resizeGambar = cv2.resize(img, (400, 500))
    # cv2.imshow('resize gambar', resizeGambar)

    cropGambar = resizeGambar[120:450, 50:370]
    # cv2.imshow('crop gambar', cropGambar)
    
    contrast = 1.3
    brightness = 15
    gambar = cv2.addWeighted(cropGambar, contrast, cropGambar, 0, brightness)
    # cv2.imshow('penajaman gambar', gambar)
    
    bilateral = cv2.bilateralFilter(gambar, 9, 75, 75)
    # cv2.imshow('smoothing gambar', bilateral)
    
    return bilateral


# ---------------------------- SEGMENTASI ----------------------------

# Batas bawah dan atas untuk warna merah dalam ruang warna HSV untuk segmentasi
lower_red_1 = np.array([0, 120, 70])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 120, 70])
upper_red_2 = np.array([180, 255, 255])

# Batas bawah dan atas untuk warna hijau dalam ruang warna HSV untuk segmentasi
lower_green = np.array([36, 100, 100])
upper_green = np.array([86, 255, 255])

# diubah ke bentuk warna HSV dan masking
def toHSV(img):
    hsvGambar = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsvGambar) 
    mask_red1 = cv2.inRange(hsvGambar, lower_red_1, upper_red_1)
    mask_red2 = cv2.inRange(hsvGambar, lower_red_2, upper_red_2)
    mask_red = mask_red1 | mask_red2
    mask_green = cv2.inRange(hsvGambar, lower_green, upper_green)
    return mask_red, mask_green

# Fungsi ini menghitung persentase piksel merah dan hijau dalam gambar berdasarkan masker yang dihasilkan dari fungsi toHSV.
def hitungPersentaseWarna(mask):
    jumlah_piksel_warna = np.count_nonzero(mask)
    total_piksel = mask.size
    persentase_warna = (jumlah_piksel_warna / total_piksel) * 100
    return persentase_warna

# Fungsi ini menggabungkan semua langkah sebelumnya: preprocessing gambar, konversi ke HSV dan masking,
# serta menghitung persentase warna merah dan hijau
def ekstraksiFitur(gambar):
    img = preprocess(gambar)
    mask_red, mask_green = toHSV(img)
    gambarHasil = cv2.bitwise_and(img, img, mask = mask_red)
    # cv2.imshow('mask', gambarHasil)
    persentase_merah = hitungPersentaseWarna(mask_red)
    persentase_hijau = hitungPersentaseWarna(mask_green)
    return persentase_merah, persentase_hijau, img, mask_red, mask_green


# ---------------------------- KLASIFIKASI ----------------------------

# menentukan kematangan berdasarkan persentase warna
def klasifikasiKematangan(gambar):
    persentase_merah, persentase_hijau, processed_img, mask_red, mask_green = ekstraksiFitur(gambar)
    persentase_warna_array = np.array([persentase_merah]).reshape(1, -1)
    # print(persentase_warna_array)
    hasil_prediksi = model.predict(persentase_warna_array)
    if persentase_hijau > persentase_merah:
        hasil = "Belum Matang"
    else:
        hasil = "Matang" if hasil_prediksi[0] == 1 else "Belum Matang"
    return hasil, persentase_merah, persentase_hijau, processed_img, mask_red, mask_green


# Fungsi ini memuat dataset dari direktori yang diberikan, mengonversi gambar ke
# format BGR yang digunakan oleh OpenCV, dan menetapkan label berdasarkan kelas ('matang' atau 'belum_matang')
def muatDataset(direktori):
    gambar = []
    label = []
    for kelas in ['matang', 'belum_matang']:
        jalur_train = f"{direktori}/train/{kelas}/*.jpeg"
        jalur_test = f"{direktori}/test/{kelas}/*.jpeg"
        koleksi_train = imread_collection(jalur_train)
        koleksi_test = imread_collection(jalur_test)
        for img in koleksi_train:
            gambar.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            label.append(1 if kelas == 'matang' else 0)
        for img in koleksi_test:
            gambar.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            label.append(1 if kelas == 'matang' else 0)
    return gambar, label

direktori_dataset = 'dataset'
gambar, label = muatDataset(direktori_dataset)
fitur = np.array([ekstraksiFitur(img)[0] for img in gambar])
fitur = fitur.reshape(-1, 1)
label = np.array(label)
X_train, X_test, y_train, y_test = train_test_split(fitur, label, test_size=0.3, random_state=42)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ---------------------------- MENAMPILKAN HASIL ----------------------------
def tampilkanHasil(gambar, hasil, persentase_merah, persentase_hijau, processed_img, mask_red, mask_green):
    img_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Resize image
    max_size = (200, 200)
    img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(img_pil)
    panel_asli.config(image=img_tk)
    panel_asli.image = img_tk

    # Display processed image
    img_processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    img_processed_pil = Image.fromarray(img_processed_rgb)
    img_processed_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
    img_processed_tk = ImageTk.PhotoImage(img_processed_pil)
    panel_processed.config(image=img_processed_tk)
    panel_processed.image = img_processed_tk

    # Determine which mask to display
    if persentase_merah > persentase_hijau:
        img_mask_rgb = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB)
    else:
        img_mask_rgb = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2RGB)

    img_mask_pil = Image.fromarray(img_mask_rgb)
    img_mask_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
    img_mask_tk = ImageTk.PhotoImage(img_mask_pil)
    panel_mask.config(image=img_mask_tk)
    panel_mask.image = img_mask_tk

    label_hasil.config(text=f"Kematangan: {hasil}")
    label_persentase.config(text=f"Persentase Merah: {persentase_merah:.2f}%\nPersentase Hijau: {persentase_hijau:.2f}%")


def prosesGambar():
    file_path = 'kersen_merah.jpeg'
    if not os.path.exists(file_path):
        messagebox.showerror("Error", "Gambar tidak ditemukan!")
        return

    gambar = cv2.imread(file_path)
    if gambar is None:
        messagebox.showerror("Error", "Gagal membaca gambar!")
        return

    hasil, persentase_merah, persentase_hijau, processed_img, mask_red, mask_green = klasifikasiKematangan(gambar)
    tampilkanHasil(gambar, hasil, persentase_merah, persentase_hijau, processed_img, mask_red, mask_green)


# ---------------------------- GUI MENGGUNAKAN TKINTER ----------------------------
root = tk.Tk()
root.title("Klasifikasi Kematangan Buah Kersen")
root.geometry("800x400")
root.minsize(800, 400)  # Set minimum window size

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

panel_asli = tk.Label(frame)
panel_asli.pack(padx=10, pady=10, side="left", fill="both", expand=True)

panel_processed = tk.Label(frame)
panel_processed.pack(padx=10, pady=10, side="left", fill="both", expand=True)

panel_mask = tk.Label(frame)
panel_mask.pack(padx=10, pady=10, side="left", fill="both", expand=True)

label_hasil = tk.Label(root, text="", font=("Helvetica", 16))
label_hasil.pack(padx=10, pady=10, side="bottom", fill="both", expand=True)

label_persentase = tk.Label(root, text="", font=("Helvetica", 16))
label_persentase.pack(padx=10, pady=10, side="bottom", fill="both", expand=True)


# ---------------------------- JALANKAN PROGRAM ----------------------------
prosesGambar()

root.mainloop()
