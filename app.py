import pandas as pd
import numpy as np
from flask import Flask, render_template, request, session, send_file
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Diperlukan untuk session

# Load model dan preprocessor (baseline ML klasik untuk perbandingan dengan quantum)
model = joblib.load('stack_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Riwayat data pasien (list sederhana untuk demo; gunakan DB untuk real)
patient_data = []

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Ambil input dari form (semua 54 fitur dari dataset CAD.csv)
        data = {
            'Age': [int(request.form['Age'])],
            'Weight': [float(request.form['Weight'])],
            'Length': [float(request.form['Length'])],
            'Sex': [request.form['Sex']],
            'BMI': [float(request.form['BMI'])],
            'DM': [int(request.form['DM'])],
            'HTN': [int(request.form['HTN'])],
            'Current Smoker': [int(request.form['Current Smoker'])],
            'EX-Smoker': [int(request.form['EX-Smoker'])],
            'FH': [int(request.form['FH'])],
            'Obesity': [request.form['Obesity']],
            'CRF': [request.form['CRF']],
            'CVA': [request.form['CVA']],
            'Airway disease': [request.form['Airway disease']],
            'Thyroid Disease': [request.form['Thyroid Disease']],
            'CHF': [request.form['CHF']],
            'DLP': [request.form['DLP']],
            'BP': [float(request.form['BP'])],
            'PR': [int(request.form['PR'])],
            'Edema': [int(request.form['Edema'])],
            'Weak Peripheral Pulse': [request.form['Weak Peripheral Pulse']],
            'Lung rales': [request.form['Lung rales']],
            'Systolic Murmur': [request.form['Systolic Murmur']],
            'Diastolic Murmur': [request.form['Diastolic Murmur']],
            'Typical Chest Pain': [int(request.form['Typical Chest Pain'])],
            'Dyspnea': [request.form['Dyspnea']],
            'Function Class': [int(request.form['Function Class'])],
            'Atypical': [request.form['Atypical']],
            'Nonanginal': [request.form['Nonanginal']],
            'Exertional CP': [request.form['Exertional CP']],
            'LowTH Ang': [request.form['LowTH Ang']],
            'Q Wave': [int(request.form['Q Wave'])],
            'St Elevation': [int(request.form['St Elevation'])],
            'St Depression': [int(request.form['St Depression'])],
            'Tinversion': [int(request.form['Tinversion'])],
            'LVH': [request.form['LVH']],
            'Poor R Progression': [request.form['Poor R Progression']],
            'FBS': [float(request.form['FBS'])],
            'CR': [float(request.form['CR'])],
            'TG': [float(request.form['TG'])],
            'LDL': [float(request.form['LDL'])],
            'HDL': [float(request.form['HDL'])],
            'BUN': [float(request.form['BUN'])],
            'ESR': [float(request.form['ESR'])],
            'HB': [float(request.form['HB'])],
            'K': [float(request.form['K'])],
            'Na': [float(request.form['Na'])],
            'WBC': [int(request.form['WBC'])],
            'Lymph': [int(request.form['Lymph'])],
            'Neut': [int(request.form['Neut'])],
            'PLT': [int(request.form['PLT'])],
            'EF-TTE': [int(request.form['EF-TTE'])],
            'Region RWMA': [int(request.form['Region RWMA'])],
            'VHD': [request.form['VHD']]
        }

        # Buat DataFrame dari input
        input_df = pd.DataFrame(data)

        # Preprocess input
        input_processed = preprocessor.transform(input_df)

        # Prediksi (baseline klasik)
        prediction = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][0] * 100  # Prob 'Cad' (kelas 0)

        # Hasil
        result = 'Cad (Penyakit Arteri Koroner)' if prediction == 0 else 'Normal'

        # Simpan ke session untuk PDF
        session['result'] = result
        session['prob'] = prob

        # Simpan ke riwayat data pasien untuk halaman Data Pasien
        patient_data.append({**data, 'Prediction': result, 'Prob Cad': prob})

        # Render ke halaman result terpisah jika ada, atau index
        return render_template('result.html')  # Ganti ke 'index.html' jika tidak ada result terpisah

    return render_template('index.html')

@app.route('/data_pasien')
def data_pasien():
    return render_template('data_pasien.html', patient_data=patient_data)

@app.route('/download_pdf')
def download_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Hasil Prediksi CAD")
    c.drawString(100, 730, f"Prediksi: {session.get('result', 'Tidak diketahui')}")
    c.drawString(100, 710, f"Probabilitas Cad: {session.get('prob', 0):.2f}%")
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="hasil_prediksi_cad.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)