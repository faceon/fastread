from fastai.text.all import *
import streamlit as st

# Streamlit launches a web app
st.markdown("# Readability Test")

class Fastread:
    def __init__(self, model_file):
        # Load model file into a learner
        self.learner = load_learner(model_file)

        # Load a text input form
        self.text = self.get_text()

        # Show readability score
        self.show_pred()

    def get_text(self):
        return st.text_input(
            label='Input text to rate readability score',
            max_chars=300,
        )
    
    def show_pred(self):
        if self.text:
            pred = self.learner.predict(self.text)[0][0]
            st.write(f'Readability score: {pred}')


if __name__ == '__main__':
    model_file = 'model_lstm.pkl'  # TODO: to be parameterized
    fastread = Fastread(model_file)