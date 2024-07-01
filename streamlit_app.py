import streamlit as st
import numpy as np
import pickle

# Modeli yükleme
def load_model():
    with open('lightgbm_titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit başlığı ve açıklaması
st.title('Titanic Hayatta Kalma Tahmini')
st.write('Aşağıdaki bilgileri girerek Titanic gemisinde hayatta kalma olasılığınızı tahmin edebilirsiniz.')

# Giriş alanları
Pclass = st.selectbox('Pclass (Bilet Sınıfı)', [1, 2, 3])
Sex = st.selectbox('Sex (Cinsiyet)', ['male', 'female'])
SibSp = st.number_input('SibSp (Kardeş/Equyaşmak)', min_value=0, max_value=10, value=0)
Parch = st.number_input('Parch (Ebeveyn/Çocuk)', min_value=0, max_value=10, value=0)
Fare = st.number_input('Fare (Bilet Ücreti)', min_value=0.0, value=50.0)
Embarked = st.selectbox('Embarked (Biniş Noktası)', ['C', 'Q', 'S'])

# Cinsiyeti sayısal değere çevirme
Sex = 1 if Sex == 'male' else 0

# Biniş Noktasını sayısal değere çevirme
Embarked_C = 1 if Embarked == 'C' else 0
Embarked_Q = 1 if Embarked == 'Q' else 0
Embarked_S = 1 if Embarked == 'S' else 0

# Tahmin butonu
if st.button('Predict'):
    input_data = np.array([[Pclass, Sex, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S]])
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)
    st.write(f'Hayatta kalma olasılığınız: %{prediction_prob[0][1] * 100:.2f}')
    if prediction[0] == 1:
        st.write("Tahmin: Hayatta kalacaksınız!")
    else:
        st.write("Tahmin: Hayatta kalamayacaksınız.")
