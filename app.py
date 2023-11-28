# Model clustering
import pickle
import numpy as np
import sklearn
knn = pickle.load(open("./static/models/model.pkl","rb"))

def trafficPrediction(data):
    data = np.array(data).reshape(1,-1)
    # Get the prediction
    result = knn.predict(data)
    return result

from flask import Flask, render_template, request

app = Flask(__name__)

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# route 'form' set as 'form.html'
@app.route('/form')
def form():
    return render_template('form.html')

# Route untuk menangani submit form
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Mengambil data dari form
        day = request.form['day']
        time = request.form['time']
        car_count = request.form['car']
        bike_count = request.form['bike']
        bus_count = request.form['bus']
        truck_count = request.form['truck']

        # Lakukan apa pun yang perlu dilakukan dengan data yang Anda terima dari form
        # Misalnya, simpan ke database, lakukan prediksi, dll.

        # Kembalikan response atau tampilkan informasi yang diperlukan
        return f'Day: {day}, Time: {time}, Car Count: {car_count}, Bike Count: {bike_count}, Bus Count: {bus_count}, Truck Count: {truck_count}'

if __name__ == '__main__':
    app.run(debug=True)
