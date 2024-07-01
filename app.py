import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
titanic_model = pickle.load(open('lightgbm_titanic_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict_api", methods=["POST"])
def predict_api():
    # Tüm JSON verisini al
    data = request.get_json()
    print(f"postman'dan gelen data: {data}")
    # "data" anahtarına erişim
    values_dict = data.get('data', {})
    print(f"data'nın anahtarına erişim: {values_dict}")
    # DataFrame'e dönüştür
    df = pd.DataFrame([values_dict])
    print(f"dataframe'e dönüştür: {df}")
    # NaN veya None değerleri kontrol et
    if df.isnull().values.any():
        nan_info = df.isnull().sum()
        return jsonify({'error': 'Input data contains NaN values', 'nan_info': nan_info.to_dict()})
    
    # DataFrame'i numpy array'e dönüştür ve yeniden şekillendir
    input_array = df.to_numpy().reshape(1, -1)
    print(f"dataframe'i numpay array'e dönüştür: {input_array}")
    
    # Tahmin yap
    prediction = titanic_model.predict(input_array)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=titanic_model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Titanic prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
   
     