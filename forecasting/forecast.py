from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Muat data yang sama
file_path = "dataset/Padi.csv"
data = pd.read_csv(file_path)

# Parameter untuk sliding window
m_past = 3  # Jumlah tahun yang digunakan sebagai input

# Fungsi untuk prediksi 5 tahun ke depan
def predict_future(provinsi):
    model_filename = f"models/ridge_model_{provinsi}.joblib"
    loaded_model = joblib.load(model_filename)

    # Ambil data untuk provinsi ini
    df_provinsi = data[data['Provinsi'] == provinsi].sort_values(by="Tahun")

    # Membuat sliding window (3 tahun)
    produksi = df_provinsi['Produksi'].values
    curah_hujan = df_provinsi['Curah hujan'].values
    luas_panen = df_provinsi['Luas Panen'].values
    suhu_rata = df_provinsi['Suhu rata-rata'].values

    sliding_windows = []
    for i in range(len(produksi) - m_past):
        window = produksi[i:i + m_past]
        window_curah = curah_hujan[i:i + m_past]
        window_luas = luas_panen[i:i + m_past]
        window_suhu = suhu_rata[i:i + m_past]
        date = df_provinsi['Tahun'].iloc[i + m_past]
        produksi_now = produksi[i + m_past]

        sliding_windows.append(
            [date] + list(window) + list(window_curah) + list(window_luas) + list(window_suhu) + [produksi_now]
        )

    # Buat DataFrame sliding windows
    feature_columns = (
        [f'Produksi-{j}' for j in range(m_past, 0, -1)] +
        [f'CurahHujan-{j}' for j in range(m_past, 0, -1)] +
        [f'LuasPanen-{j}' for j in range(m_past, 0, -1)] +
        [f'SuhuRata-{j}' for j in range(m_past, 0, -1)]
    )
    sliding_windows_dt = pd.DataFrame(
        sliding_windows,
        columns=['Year'] + feature_columns + ['Produksi Now']
    )

    # Normalisasi data
    features = feature_columns
    target = 'Produksi Now'
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    normalized_features = feature_scaler.fit_transform(sliding_windows_dt[features])
    normalized_target = target_scaler.fit_transform(sliding_windows_dt[[target]])

    normalized_df = pd.DataFrame(normalized_features, columns=features)
    normalized_df[target] = normalized_target
    normalized_df['Year'] = sliding_windows_dt['Year']

    # Data terakhir untuk prediksi
    last_data = normalized_df[features].iloc[-1].values.reshape(1, -1)

    # Prediksi untuk 5 tahun ke depan
    years_to_predict = 5
    future_years = []
    future_productions = []
    current_window = last_data

    for i in range(years_to_predict):
        current_window_df = pd.DataFrame(current_window, columns=features)

        # Prediksi tahun berikutnya
        next_pred = loaded_model.predict(current_window_df)

        # Inversi prediksi ke skala asli
        pred_ori = target_scaler.inverse_transform(next_pred.reshape(-1, 1))
        predicted_year = normalized_df['Year'].iloc[-1] + i + 1

        # Simpan hasil prediksi
        future_years.append(predicted_year)
        future_productions.append(pred_ori[0, 0])

        # Update sliding window dengan prediksi terbaru
        next_pred = next_pred.reshape(1, -1)
        current_window = np.concatenate((current_window[:, 1:], next_pred), axis=1)

    predicted_df = pd.DataFrame({
        'Year': future_years,
        'Produksi': future_productions
    })

    # Visualisasi hasil prediksi
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(normalized_df['Year'], target_scaler.inverse_transform(normalized_df[target].values.reshape(-1, 1)), label='Actual', color='blue')
    ax.scatter(predicted_df['Year'], predicted_df['Produksi'], label='Future Predictions (5 years)', color='green', s=100)
    ax.set_title(f'Produksi Aktual dan Prediksi 5 Tahun ke Depan ({provinsi})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Produksi (Original Scale)')
    ax.set_xticks(range(normalized_df['Year'].min(), predicted_df['Year'].max() + 1, 2))
    ax.set_xticklabels(range(normalized_df['Year'].min(), predicted_df['Year'].max() + 1, 2), rotation=45)
    ax.legend()
    ax.grid(True)

    # Save plot to a PNG image in memory
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    canvas.print_png(img)
    img.seek(0)
    img_b64 = base64.b64encode(img.read()).decode('utf-8')

    return img_b64, predicted_df

@app.route('/')
def index():
    # Siapkan data untuk semua provinsi
    predictions = {}
    provinces = data['Provinsi'].unique()

    for provinsi in provinces:
        try:
            # Buat prediksi untuk masing-masing provinsi
            img_b64, predicted_df = predict_future(provinsi)
            
            # Membulatkan nilai prediksi pada kolom 'Produksi'
            predicted_df['Produksi'] = predicted_df['Produksi'].round(0).astype(int)

            predictions[provinsi] = {
                'image': img_b64,
                'data': predicted_df.to_dict(orient='records')
            }
        except Exception as e:
            print(f"Error predicting for {provinsi}: {e}")
            predictions[provinsi] = {
                'image': None,
                'data': []
            }

    # Kirim data provinsi dan hasil prediksi ke template
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
