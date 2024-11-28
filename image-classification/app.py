from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memuat model
MODEL_PATH = 'model/lvq-model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Daftar nama kelas
class_names = ['Leafcurl', 'Yellowwish', 'healthy']

# Fungsi untuk memproses gambar
def load_and_process_image(image_path, target_size=(100, 100)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten() / 255.0

# Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Periksa apakah file telah diunggah
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Lakukan prediksi
            processed_image = load_and_process_image(file_path)
            processed_image = processed_image.reshape(1, -1)  # Tambahkan dimensi batch
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_index]

            # Tampilkan hasil prediksi pada halaman hasil
            return render_template('result.html', image_url=file_path, prediction=predicted_class)
    
    return render_template('index.html')

# Jalankan aplikasi Flask
if __name__ == '__main__':
    # Pastikan folder upload ada
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)