from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import re
import string

app = Flask(__name__)

# Daftar kata positif dan negatif
positive_words = ['baik', 'indah', 'bagus', 'menyenangkan', 'luar biasa', 'puas', 'ramah', 'bersih']
negative_words = ['buruk', 'kotor', 'mengecewakan', 'tidak puas', 'jelek', 'kasar', 'mahal']

# Preprocessing teks
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text.lower())
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Fungsi untuk menentukan sentimen
def predict_sentiment(sentence):
    processed_sentence = preprocess_text(sentence)
    positive_count = sum(word in processed_sentence for word in positive_words)
    negative_count = sum(word in processed_sentence for word in negative_words)

    if positive_count > negative_count:
        return "Bermanfaat"
    elif negative_count > positive_count:
        return "Tidak Bermanfaat"
    else:
        return "Tidak Bermanfaat"  # Jika sama, anggap sebagai Tidak Bermanfaat

# Fungsi untuk memproses file CSV
def process_csv(file):
    try:
        # Membaca file dengan encoding 'utf-8'
        df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        # Jika gagal, gunakan encoding alternatif 'latin-1'
        df = pd.read_csv(file, encoding='latin-1', on_bad_lines='skip')
    except pd.errors.ParserError as e:
        return None, None, f"Kesalahan parsing file CSV: {e}"

    if 'Ulasan' in df.columns:
        df['Ulasan'] = df['Ulasan'].fillna("")
        df['processed_content'] = df['Ulasan'].apply(preprocess_text)
        df['Sentimen Prediksi'] = df['processed_content'].apply(predict_sentiment)

        total_reviews = len(df)
        positive_reviews = len(df[df['Sentimen Prediksi'] == 'Bermanfaat'])
        negative_reviews = len(df[df['Sentimen Prediksi'] == 'Tidak Bermanfaat'])

        positive_percentage = (positive_reviews / total_reviews) * 100
        negative_percentage = (negative_reviews / total_reviews) * 100
        return positive_percentage, negative_percentage, df
    else:
        return None, None, 'Kolom "Ulasan" tidak ditemukan.'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ulasan = request.form['ulasan']
        if ulasan.strip():
            sentiment = predict_sentiment(ulasan)
            return render_template('result.html', ulasan=ulasan, sentiment=sentiment)
        else:
            return redirect(url_for('home', error="Harap masukkan ulasan yang valid."))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return redirect(url_for('home', error="Harap unggah file CSV."))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home', error="Harap unggah file CSV yang valid."))
    
    positive_percentage, negative_percentage, processed_df = process_csv(file)
    if positive_percentage is not None:
        result_csv = processed_df.to_csv(index=False)
        return render_template('batch_result.html', 
                               positive_percentage=positive_percentage, 
                               negative_percentage=negative_percentage,
                               result_csv=result_csv)
    elif isinstance(processed_df, str):
        return redirect(url_for('home', error=processed_df))
    else:
        return redirect(url_for('home', error='Kolom "Ulasan" tidak ditemukan dalam file CSV yang diunggah.'))

if __name__ == '__main__':
    app.run(debug=True)
