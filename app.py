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


############################################################################

# app.py
from flask import Flask, request, render_template # Import flask libraries

# Initialize the flask class and specify the templates directory
app = Flask(__name__)

# Default route set as 'index'
@app.route('/')
def index():
    return render_template('index.html')

# route 'form' set as 'form.html'
@app.route('/form')
def form():
    return render_template('form.html')

# Route 'result' accepts POST request
@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        # Mengambil data dari form
        day = request.form['day']
        time = request.form['time']
        car_count = request.form['car']
        bike_count = request.form['bike']
        bus_count = request.form['bus']
        truck_count = request.form['truck']

        # Get the prediction from the model
        result = customerClustering(user_input)
        if(result == 0):
            result = 'Low'
        elif(result == 1):
            result = 'Normal'
        elif(result == 2):
            result = 'High'
        elif(result == 3):
            result = 'Heavy'
        else:
            result = 'Unknown'
        return f'Day: {day}, Time: {time}, Car Count: {car_count}, Bike Count: {bike_count}, Bus Count: {bus_count}, Truck Count: {truck_count}'        
    
# Run the Flask server
if(__name__=='__main__'):
    app.run()