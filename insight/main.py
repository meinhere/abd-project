import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from clustering import *
from image_classification import load_and_process_image, classify_image
from text_classification import predict_sentiment, process_csv
from forecasting import predict_future
from plot import create_cluster_plot, create_map

app = Flask(__name__)

# Load data mpp
mpp_df = pd.read_csv('dataset/mpp.csv', delimiter=",")
mpp_df = preprocess_mpp_data(mpp_df)

# Load data padi
padi_df = pd.read_csv('dataset/padi.csv', delimiter=",")

@app.route("/", methods=["GET", "POST"])
def index():

    # =================== Clustering =====================
    available_columns = mpp_df.columns.tolist()[1:]
    selected_columns = request.form.getlist("columns") if request.method == "POST" else ['2021']
    n_clusters = int(request.form.get("n_clusters", 3))

    features, features_scaled = perform_kmeans(mpp_df, selected_columns)
    features['Cluster'] = cluster_data(features_scaled, n_clusters)

    # =================== Forecasting =====================
    predictions = {}
    provinces = mpp_df['Provinsi'].unique()

    for provinsi in provinces:
        try:
            # Buat prediksi untuk masing-masing provinsi
            img_b64, predicted_df = predict_future(padi_df, provinsi)
            # Membulatkan nilai prediksi pada kolom 'Produksi'
            predicted_df['Produksi'] = predicted_df['Produksi'].round(0).astype(int)

            predictions[provinsi] = {
                'image': img_b64,
                'data': predicted_df.to_dict(orient='records')
            }
        except Exception as e:
            # print(f"Error predicting for {provinsi}: {e}")
            predictions[provinsi] = {
                'image': None,
                'data': []
            }

    # Generate map
    map_html = create_map(mpp_df['Provinsi'], features, predictions)

    return render_template("index.html", available_columns=available_columns, selected_columns=selected_columns, map_html=map_html, n_clusters=n_clusters)

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/klasifikasi-gambar", methods=["GET", "POST"])
def klasifikasi_gambar():
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
            predicted_class = classify_image(processed_image)

            # Tampilkan hasil prediksi pada halaman hasil
            return render_template('image_result.html', image_url=file_path, prediction=predicted_class)
    
    return render_template('image.html')

@app.route("/klasifikasi-teks", methods=["GET", "POST"])
def klasifikasi_teks():
    if request.method == 'POST':
        # Ambil teks dari form
        ulasan = request.form['ulasan']
        if ulasan.strip():
            # Lakukan prediksi
            sentiment = predict_sentiment(ulasan)
            return render_template('text_result.html', ulasan=ulasan, sentiment=sentiment)
        else:
            return redirect(url_for('klasifikasi_teks', error="Harap masukkan teks yang valid."))
    
    return render_template('text.html')

@app.route("/klasifikasi-batch", methods=["POST"])
def klasifikasi_batch():
    if 'file' not in request.files:
        return redirect(url_for('klasifikasi_teks', error="Harap unggah file CSV."))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('klasifikasi_teks', error="Harap unggah file CSV yang valid."))
    
    positive_percentage, negative_percentage, processed_df = process_csv(file)
    # print(processed_df)
    if positive_percentage is not None:
        result_csv = processed_df.to_csv(index=False)
        return render_template('text_result.html', 
                               positive_percentage=positive_percentage, 
                               negative_percentage=negative_percentage,
                               result_csv=result_csv)
    elif isinstance(processed_df, str):
        return redirect(url_for('klasifikasi_teks', error=processed_df))
    else:
        return redirect(url_for('klasifikasi_teks', error='Kolom "Ulasan" tidak ditemukan dalam file CSV yang diunggah.'))

if __name__ == "__main__":
    app.run(debug=True)