# Importing Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create a title and sub-title
st.title("Diabetes Prediction")
st.write("""
Detect if someone has diabetes using machine learning and python!
""")

#open and display an image
image = Image.open('diabetes.png')
st.image(image, caption = 'ML', use_column_width = True)

#get the data
df = pd.read_csv('diabetes.csv')

#set a subheader
st.subheader('Data information:')

#show the data as a table
st.dataframe(df)

st.subheader('Data Description:')
#show statistics on the data
st.write(df.describe())

st.subheader('Data Visualization:')
#show the data as a chart
chart = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y'
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#split the dataset into 75% training and 25% testing
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size=0.25, random_state=0)

#get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 3)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.0, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.03725)
    age = st.sidebar.slider('age', 21, 81, 29)

    #store a dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness':skin_thickness,
        'insulin': insulin,
        'BMI':  BMI,
        'DPF': DPF,
        'age':age
    }

    #transform the data into a dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

#store the user input into a variable
user_input = get_user_input()

#set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#show the model metrica
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

#Store the models predictions in avariable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification:')
st.write(prediction)
st.subheader('Result: ')
if(prediction==1):
    st.write('''# You have Diabetes.''')
    st.write('''You must take the following medicines: ''')
    st.write('''1.Insulin (long- and rapid-acting)''')
    st.write('''2.Metformin (biguanide class)''')
    st.write('''3.Glipizide (sulfonylurea class)''')
    st.write('''4.Glimepiride (sulfonylurea class)''')
    st.write('''5.Invokana (sodium glucose cotransporter 2 inhibitor class)''')
    st.write('''6.Jardiance (SGLT2 class)''')
    st.write('''7.Januvia (dipeptidyl peptidase 4 inhibitor)''')
    st.write('''8.Pioglitazone (thiazolidinediones)''')
    st.write('''9.Victoza (glucagon-like peptide 1 agonist)''')
    st.write('''10.Trulicity (glucagon-like peptide 1 agonist)''')
    
else:
    st.write('''You do not have diabetes.''')






