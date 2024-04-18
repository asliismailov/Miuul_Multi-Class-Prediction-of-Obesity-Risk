import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the model using joblib
model = joblib.load("Miuul_Final.pkl")

def display_about():
    st.title('About Us')
    st.write('''
    This application is developed to predict obesity status based on users' health data.
    It analyzes various dietary and lifestyle data to assess the risk of obesity.
    Our project aims to provide informative content to help users better understand and manage their health.
    ''')

    st.subheader('Team Members')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('asli.jpeg', width=150)
        st.markdown('**Aslı Öztürk**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/ozturk-asli/)')
        st.write('Aslı is an expert researcher in data science and machine learning. She has worked with large datasets in various projects and gained experience in health technologies.')

    with col2:
        st.image('begum.jpeg', width=150)
        st.markdown('**Begüm Baybora**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/begumbaybora/)')
        st.write('Begüm is a software engineer with extensive experience in artificial intelligence and data analysis. She has developed solutions in educational technologies and the health sector.')

    with col3:
        st.image('ugur.jpeg', width=150)
        st.markdown('**Uğur Can Odabaşı**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/ugurcanodabasi)')
        st.write('Uğur Can is deeply knowledgeable in software development and systems engineering. He has provided technology solutions across various industries and served as a technical leader.')

# Homepage
def home_page():
    st.title('Homepage')

    # Display a world map of obesity rates (placeholder static image)
    st.image('Obesity_rate_(WHO,_2022).png', caption='World Obesity Map')

    # Text describing obesity as the disease of the era
    st.write("""
    ## Obesity: The Disease of Our Era

    Obesity is a serious health issue affecting millions worldwide and spreading rapidly.
    Individuals with a body mass index (BMI) over 30 are classified as obese, which significantly increases the risk of various chronic diseases.
    Obesity can lead to health issues like heart disease, type 2 diabetes, some types of cancer, and musculoskeletal disorders.

    Combating obesity requires not only individual efforts but also global cooperation.
    Adopting healthy dietary habits, engaging in regular physical activity, and making healthy lifestyle choices play a key role in fighting obesity.
    Communities and governments should develop and implement policies to facilitate access to healthy foods, promote physical activity, and enhance health education.
    """)

#Görseller
def graph_page():
    col1,col2 = st.columns()
with col1:
   st.header("Features")
   st.image("features.jpeg")

with col2:
   st.header("Learning Curve")
   st.image("learning_curve.png")

# Predict Obesity
def predict_obesity():
    st.title('Predict Obesity Status')

    # Collect user input
    gender = st.selectbox('Your Gender', ['Male', 'Female'])
    age = st.number_input('Your Age', min_value=1, max_value=100, value=25)
    height = st.number_input('Your Height (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Your Weight (kg)', min_value=20, max_value=200, value=70)
    family_history = st.selectbox('Family history of obesity?', ['Yes', 'No'])
    favc = st.selectbox('Frequency of high caloric food consumption?', ['Never', 'Occasionally', 'Frequently'])
    fcvc = st.selectbox('Frequency of vegetable consumption?', ['Never', 'Occasionally', 'Frequently', 'Very Frequently'])
    ncp = st.selectbox('Number of main meals per day?', ['1', '2', '3', '4 or more'])
    caec = st.selectbox('Eating behavior?', ['Never', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox('Do you smoke?', ['Yes', 'No'])
    ch2o = st.slider('Daily water intake (liters)', 0.0, 5.0, 2.0)
    scc = st.selectbox('Do you monitor your calorie intake?', ['Yes', 'No'])
    faf = st.slider('Weekly physical activity frequency (days)', 0, 7, 3)
    tue = st.slider('Daily technology usage (hours)', 0, 24, 8)
    calc = st.selectbox('Alcohol consumption frequency?', ['Never', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox('Main mode of transportation?', ['Walking', 'Bicycle', 'Public Transportation', 'Car', 'Motorcycle'])

    # Convert categorical data to numerical
    gender = 1 if gender == 'Male' else 0
    family_history = 1 if family_history == 'Yes' else 0
    favc = 1 if favc == 'Frequently' else 0
    fcvc_mapping = {'Never': 0, 'Occasionally': 1, 'Frequently': 2, 'Very Frequently': 3}
    fcvc = fcvc_mapping[fcvc]
    ncp_mapping = {'1': 1, '2': 2, '3': 3, '4 or more': 4}
    ncp = ncp_mapping[ncp]
    caec_mapping = {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    caec = caec_mapping[caec]
    smoke = 1 if smoke == 'Yes' else 0
    scc = 1 if scc == 'Yes' else 0
    calc_mapping = {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc = calc_mapping[calc]
    mtrans_mapping = {'Walking': 0, 'Bicycle': 1, 'Public Transportation': 2, 'Car': 3, 'Motorcycle': 4}
    mtrans = mtrans_mapping[mtrans]

    # Construct the input vector including all variables
    input_data = np.array(
        [gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans])

    if st.button('Predict'):
        prediction = model.predict(input_data.reshape(1, -1))
        # Use a dictionary to explain the obesity status
        obesity_status = {
            0: 'Insufficient Weight',
            1: 'Normal Weight',
            2: 'Overweight Level I',
            3: 'Overweight Level II',
            4: 'Type I Obesity',
            5: 'Type II Obesity',
            6: 'Type III Obesity'
        }
        predicted_status = obesity_status.get(prediction[0], "Unknown status")
        st.subheader(f'Predicted Obesity Status: {predicted_status}')

        # Add visual and recommendation text
        advice_dict = {
            'Insufficient Weight': ("It is important to eat regularly and consume quality nutrients. Focus on increasing your calorie intake."),
            'Normal Weight': ("Continue your current healthy lifestyle and regular physical activities."),
            'Overweight Level I': ("Consider starting a regular exercise regime and healthy diet plan to lose weight."),
            'Overweight Level II': ("Consult health professionals to prepare a personalized weight loss plan."),
            'Type I Obesity': ("Review your eating and exercise habits, consider getting professional help if necessary."),
            'Type II Obesity': ("Seek immediate medical attention for serious health risks."),
            'Type III Obesity': ("Seek immediate medical help and adopt a multidisciplinary approach to combat obesity.")
        }
        advice = advice_dict[predicted_status]
        st.write(advice)

# Other functions and Streamlit configuration

# Configure the Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select Page:', ['About Us', 'Homepage', 'Dynamic Graphs', 'Predict Obesity'])

    if page == 'About Us':
        display_about()
    elif page == 'Homepage':
        home_page()
    elif page == 'Görseller':
        graph_page()
    elif page == 'Predict Obesity':
        predict_obesity()

if __name__ == "__main__":
    main()

