import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv("diabetes.csv")

st.title("Diabetes Checkup")

st.subheader("Training Data")
st.write(df.describe())

st.subheader("Visualization")
st.bar_chart(df)

X = df.drop(['Outcome'],axis=1)
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)    # min, max, starting_value
    glucose = st.sidebar.slider('Glucose', 0, 199, 120)
    bloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 70)
    skinThickness = st.sidebar.slider('SkinThickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 0, 81, 33)

    user_report = {
        'Pregnancies' : pregnancies,
        'Glucose' : glucose,
        'BloodPressure' : bloodPressure,
        'SkinThickness' : skinThickness,
        'Insulin' : insulin,
        'BMI' : bmi,
        'DiabetesPedigreeFunction' : dpf,
        'Age' : age,
    }

    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()

# Create a bar chart
def create_diabetes_distribution_chart(df):
    diabetes_counts = df['Outcome'].value_counts()
    fig, ax = plt.subplots()
    diabetes_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Diabetic vs Non-Diabetic Patients')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Non-Diabetic', 'Diabetic'], rotation=0)
    return fig

# Streamlit app
st.subheader('Diabetes Bar Chart')

st.write("This app shows the distribution of diabetic and non-diabetic patients in the dataset.")

# Display the bar chart
st.pyplot(create_diabetes_distribution_chart(df))

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.subheader("Accuracy")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

user_result = model.predict(user_data)
st.subheader("Your Report: ")

output = ""
if user_result[0] == 0:
    output = "Congratulations! You are healthy."
else:
    output = "I'm regret to inform that you are not healthy"

st.write(output)