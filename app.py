from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Import model
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    lap_type = request.form['lap_type']
    ram_type = int(request.form['ram_type'])
    weight = float(request.form['weight'])
    touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
    ips = 1 if request.form['ips'] == 'Yes' else 0
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    hdd = int(request.form['hdd'])
    ssd = int(request.form['ssd'])
    gpu = request.form['gpu']
    os = request.form['os']

    X_res, Y_res = map(int, resolution.split('x'))
    if screen_size != 0:
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    else:
        return render_template('index.html', error="Screen size cannot be zero.")

    query = np.array([company, lap_type, ram_type, weight, touchscreen, ips,
                      ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    try:
        prediction = pipe.predict(query)
        price = int(np.exp(prediction[0]))
        return render_template('index.html', prediction_text=f'The Price of laptop is ₹{price}')
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
