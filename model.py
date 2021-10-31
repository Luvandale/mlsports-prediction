import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

# from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("r_regressor.pkl","rb")
regressor=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Reactions,Composure,Potential,Wage):
    
    """Let's predict the overall
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Reactions
        in: query
        type: number
        required: true
      - name: Composure
        in: query
        type: number
        required: true
      - name: Potential
        in: query
        type: number
        required: true
      - name: Wage
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=regressor.predict([[Reactions,Composure,Potential,Wage]])
    print(prediction)
    return prediction



def main():
    st.title("Overall score predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Score predictor ML app </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Reactions = st.text_input("Reactions","Type Here")
    Composure = st.text_input("Composure","Type Here")
    Potential = st.text_input("Potential","Type Here")
    Wage= st.text_input("Wage","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Reactions,Composure,Potential,Wage)
    st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()