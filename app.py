import os
from dotenv import load_dotenv
from models import Models
import streamlit as st

load_dotenv()

hf_api = os.getenv("HF_API_KEY")

models = Models()

def main():

    st.title('Models App')
    st.write('Welcome to my Streamlit app!')

    st.sidebar.image("static/logo.jpeg", use_column_width=True)

    st.sidebar.markdown(
        """
        <h1 style="text-align: center;">Welcome to Model App</h1>
        """, unsafe_allow_html=True
    )

    text_input_zero_shot = st.text_input(
        "Enter a zero_shot_classification sequence👇"
    )

    if text_input_zero_shot:
        st.write("You entered: ", models.zero_shot_classification(text_input_zero_shot))


    text_input_trans = st.text_input(
        "Enter a translation sequence👇"
    )
    if text_input_trans:
        st.write("You entered: ", models.zero_shot_classification(text_input_trans))



if __name__ == '__main__':
    main()