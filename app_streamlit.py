import streamlit as st
import re
import pickle
import pandas as pd
import io

# ====== Preprocessing Function ======
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # hapus link
    text = re.sub(r"@\w+|#\w+", "", text)  # hapus mention dan hashtag
    text = re.sub(r"[^a-z\s]", "", text)  # hapus simbol dan angka
    return text

# ====== Load Model & Vectorizer ======
model = pickle.load(open("logistic_regression.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ====== Streamlit UI ======
st.set_page_config(page_title="Analisis Sentimen", layout="centered")
st.title("Analisis Sentimen Tweet")
st.write("Masukkan tweet, atau upload file CSV untuk prediksi massal.")

# ====== Prediksi Manual ======
st.subheader("📝 Prediksi Satu Tweet")

tweet_input = st.text_area("✍️ Masukkan tweet di sini")

if st.button("🔍 Prediksi Sentimen"):
    if tweet_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        cleaned = clean_text(tweet_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 2:
            st.success("✅ Sentimen Positif")
        elif prediction == 1:
            st.info("ℹ️ Sentimen Netral")
        else:
            st.error("❌ Sentimen Negatif")


# ====== Prediksi dari File CSV ======
st.markdown("---")
st.subheader("📁 Prediksi Sentimen dari File CSV")

uploaded_file = st.file_uploader("Unggah file CSV (dengan kolom 'Tweet')", type=["csv"])

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')

        if 'Tweet' not in df_csv.columns:
            st.error("❌ Kolom 'Tweet' tidak ditemukan di file.")
        else:
            df_csv.dropna(subset=['Tweet'], inplace=True)  # buang baris kosong

            df_csv['clean'] = df_csv['Tweet'].apply(clean_text)
            vectorized_bulk = vectorizer.transform(df_csv['clean'])
            predictions = model.predict(vectorized_bulk)

            # Mapping label angka ke teks
            label_map = {
                0: "Negatif",
                1: "Netral",
                2: "Positif"
            }

            df_csv['Sentimen'] = [label_map.get(p, "Tidak Dikenal") for p in predictions]

            st.success("✅ Prediksi selesai!")
            st.dataframe(df_csv[['Tweet', 'Sentimen']])

            # Simpan ke Excel
            to_download = df_csv[['Tweet', 'Sentimen']]
            to_excel = io.BytesIO()
            to_download.to_excel(to_excel, index=False, sheet_name='Hasil')
            to_excel.seek(0)

            st.download_button(
                label="📥 Download Hasil (.xlsx)",
                data=to_excel,
                file_name="hasil_sentimen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
