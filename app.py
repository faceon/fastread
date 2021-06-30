from fastai.text.all import *
import fastai
import streamlit as st


def get_text():
    return st.text_input(label='Input text to rate readability score',
                         max_chars=300,)


@st.cache(hash_funcs={TextLearner: (lambda x: 1)})  
# This forces get_learner to cache the model always
def get_learner(model_file):
    learner = load_learner(model_file)
    return learner


def show_pred(text, model):
    if text:
        pred = learner.predict(text)[0][0]
        st.write(f'Readability score: {pred}')


if __name__ == '__main__':
    MODEL_FILE = 'model_lstm.pkl'
    st.title("Readability Test")
    
    text = get_text()
    learner = get_learner(MODEL_FILE)
    show_pred(text, learner)