from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Niestandardowa warstwa, która usuwa argument 'groups'
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# Dodanie niestandardowej warstwy do custom_objects
get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

# Wczytanie modelu
model = load_model('keras_model.h5')

# CAMERA może być 0 lub 1 w zależności od domyślnej kamery komputera.
camera = cv2.VideoCapture(0)

# Pobranie etykiet z pliku labels.txt
with open('labels.txt', 'r') as file:
    labels = [line.strip().split(' ', 1)[1] for line in file.readlines()]

# Deklaracja cen produktów (przykłady)
product_prices = {
    "Jablko": 0.8,
    "Ibum": 9.99,
    "Herbata": 6.5,
    "Banan": 1.0,
    "Kubek": 8.0,
    "Pomidor": 1.5
}

# Inicjalizacja Tkinter
root = tk.Tk()
root.title("Rozpoznane Produkty")

# Lista produktów i podsumowanie cen
recognized_products = {}
total_price = 0.0


# Funkcja aktualizująca listę produktów i podsumowanie cen z formatowaniem
def update_product_list_formatted():
    global total_price
    product_list.delete(0, tk.END)
    total_price = 0.0
    # Ustawienie szerokości kolumn
    col_widths = {'nazwa': 20, 'ilosc': 10, 'cena': 15}
    header_format = "{:<{nazwa}}{:<{ilosc}}{:>{cena}}"
    product_format = "{:<{nazwa}}{:<{ilosc}}{:>{cena}.2f} zł"

    header = header_format.format("NAZWA PRODUKTU", "Ilość", "Cena całkowita", **col_widths)
    product_list.insert(tk.END, header)

    for product, count in recognized_products.items():
        product_line = product_format.format(product, count, product_prices[product] * count, **col_widths)
        product_list.insert(tk.END, product_line)
        total_price += product_prices[product] * count

    total_price_label.config(text=f"Suma: {total_price:.2f} zł")



# Tworzenie elementów GUI w oknie Tkinter
product_list = tk.Listbox(root, font=('Courier', 14))
product_list.pack(fill=tk.BOTH, expand=True)

total_price_label = tk.Label(root, text=f"Suma: zł{total_price:.2f}")
total_price_label.pack()

# Funkcja obsługująca dodawanie rozpoznanych produktów
def add_product(label_text):
    if label_text not in product_prices:
        messagebox.showerror("Błąd", f"Brak ceny dla produktu: {label_text}")
        return

    answer = messagebox.askyesno("Potwierdzenie", f"Czy to na pewno {label_text}?")
    if answer:
        if label_text in recognized_products:
            recognized_products[label_text] += 1
        else:
            recognized_products[label_text] = 1
        update_product_list_formatted()

# Funkcja skanowania produktu
def scan_product():
    # Przechwycenie obrazu z kamery
    ret, image = camera.read()
    if not ret:
        messagebox.showerror("Błąd", "Nie można przechwycić obrazu z kamery")
        return

    # Zmiana rozmiaru obrazu do (224, 224) pikseli
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Konwersja obrazu do tablicy numpy i normalizacja
    img_array = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img_array = (img_array / 127.5) - 1

    # Predykcja modelu
    probabilities = model.predict(img_array)

    # Wypisanie prawdopodobieństw
    print(list(probabilities[0]))

    # Wypisanie etykiety z najwyższym prawdopodobieństwem, jeśli jest większe niż 0.99
    if np.max(probabilities) >= 0.99:
        label_index = np.argmax(probabilities)
        label_text = labels[label_index]
        print(label_text)
        add_product(label_text)
    else:
        messagebox.showinfo("Brak Rozpoznania", "Nie rozpoznano żadnego przedmiotu z wysoką pewnością.")

# Funkcja wyświetlania obrazu na planszy Tkinter
def show_frame():
    ret, frame = camera.read()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, show_frame)

# Dodanie przycisku "Zeskanuj produkt"
scan_button = tk.Button(root, text="Zeskanuj produkt", command=scan_product)
scan_button.pack()

# Label do wyświetlania obrazu z kamerki
camera_label = tk.Label(root)
camera_label.pack()

# Rozpoczęcie wyświetlania obrazu z kamerki
show_frame()

# Główna pętla Tkinter
root.mainloop()

# Zwolnienie kamery
camera.release()
