!pip install langchain==0.0.350
!pip install streamlit==1.29.0
!pip install openai==1.28.2

import streamlit as st
from langchain.llms import OpenAI

st.set_page_config(page_title="Chat with your bot!")
st.title("Ask anythings!")

import os
# os.environ["OPENAI_API_KEY"] = "###"

def generate_response(input_text):
    llm = OpenAI(model_name='gpt-4-0314', temperature=0)
    st.info(llm(input_text))

with st.form('Question'):
    text= st.text_area('Input Question', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button('send')
    generate_response(text)
