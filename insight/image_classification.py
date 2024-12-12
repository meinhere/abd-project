import numpy as np
import tensorflow as tf
import cv2

# Fungsi untuk memproses gambar
def load_and_process_image(image_path, target_size=(100, 100)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten() / 255.0

# Fungsi untuk mengklasifikasikan gambar 
def classify_image(processed_image):
    # Memuat model
    MODEL_PATH = 'models/lvq-model.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
  
    # Daftar nama kelas
    class_names = ['Leafcurl', 'Yellowwish', 'healthy']
  
    processed_image = processed_image.reshape(1, -1)  # Tambahkan dimensi batch
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    return predicted_class