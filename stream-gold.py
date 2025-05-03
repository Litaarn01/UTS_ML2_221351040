import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Memuat model dan scaler
model = tf.lite.Interpreter(model_path="gold_price_prediction_model.tflite")
model.allocate_tensors()
scaler = joblib.load("scaler.pkl")

st.title('Prediksi Harga Emas')
future_date = st.date_input("Masukkan Tanggal Prediksi")

# Input untuk jumlah gram emas
grams = st.number_input("Masukkan Jumlah Gram Emas", min_value=1, value=1)

# Fungsi untuk mendapatkan data historis
def get_historical_data(future_date):
    return np.random.rand(60)  

def predict_price(future_date):
    historical_data = get_historical_data(future_date)
    scaled_data = scaler.transform(historical_data.reshape(-1, 1))
    input_data = scaled_data.reshape(1, 60, 1)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    model.invoke()
    predicted_price_scaled = model.get_tensor(output_details[0]['index'])
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    return predicted_price

# Fungsi untuk melihat harga naik dan turun
def get_high_low_prices(future_date):
    historical_data = get_historical_data(future_date)
    if historical_data is None or len(historical_data) == 0:
        st.error("Tidak ada data historis yang tersedia untuk menghitung harga tertinggi dan terendah.")
        return None, None
    highest_price = np.max(historical_data)
    lowest_price = np.min(historical_data)
    return highest_price, lowest_price

# Fungsi untuk format harga dalam Rupiah
def format_rupiah(amount):
    return f"Rp {amount:,.2f}"

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    st.write("Tombol Prediksi Ditekan")
    predicted_price = predict_price(future_date)
    if predicted_price is not None:
        total_price = predicted_price * grams  # Menghitung total harga berdasarkan jumlah gram
        st.success(f"Harga emas pada {future_date} diperkirakan: {format_rupiah(predicted_price)} per gram")
        st.success(f"Total harga untuk {grams} gram emas berdasarkan tanggal prediksi: {format_rupiah(total_price)}")
    else:
        st.error("Prediksi gagal. Pastikan data historis cukup.")

# Tombol untuk melihat harga naik dan turun
if st.button('Lihat Naik dan Turun Harga Emas'):
    highest_price, lowest_price = get_high_low_prices(future_date)
    if highest_price is not None and lowest_price is not None:
        st.subheader("Harga Emas Tertinggi dan Terendah (60 Hari Sebelumnya)")
        st.write(f"Kenaikan Harga Hari Ini: {format_rupiah(highest_price)}")
        st.write(f"Penurunan Harga Hari Ini: {format_rupiah(lowest_price)}")

# Tombol tambahan untuk melihat data historis
if st.button('Lihat Data Historis'):
    historical_data = get_historical_data(future_date)
    if historical_data is not None and len(historical_data) > 0:
        st.subheader("Data Historis (60 Hari Sebelumnya)")
        st.line_chart(historical_data)
