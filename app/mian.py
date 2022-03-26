from typing import List
import re
from BotEngine import BotEngine
from random import random

import streamlit as st
from streamlit_chat import message as st_message


MODEL_PATH = 'model/universal-sentence-encoder-multilingual-qa_3'
DATA_PATH = 'WHO_FAQ.xlsx'
HELP_TEXT = '''
        Commonly asked questions:\n
            1. What is a coronavirus?\n
            2. What are the symptoms of COVID-19?\n
            3. How does COVID-19 spread?\n
            4. What can I do to protect myself from this virus?\n
            5. How long does the virus survive on surfaces?\n
            6. Vaccines for COVID-19?\n
            There can be more questions. Enter your question in the text box below.
'''


st.set_page_config(
        page_title="Covid-19 Chat Bot",
        page_icon=":robot:"
)

def preprocess_sentences(input_sentences: List[str])-> List[str]:
    '''
    Preprocesses the input sentences to be compatible with the model.(The model has never seen covid-19 or covid before, so we replace it with coronavirus)
    args:
        input_sentences:list, list of sentences
    returns:
        preprocessed_sentences:list, list of preprocessed sentences
    '''
    return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I) 
            for input_sentence in input_sentences]

@st.experimental_singleton
def init_bot_engine():
    return BotEngine(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        preprocess_func=preprocess_sentences,
        help_text=HELP_TEXT,
    )
    

def generate_response()->None:
    bot = init_bot_engine()

    widget_1_key = str(random())
    widget_2_key = str(random())

    user_message = st.session_state.input_text

    if user_message.lower().strip() == "help":
        st.session_state.history.append({"message": user_message, "is_user": True, 'key': widget_1_key})
        st.session_state.history.append({"message": HELP_TEXT, "is_user": False, 'key': widget_2_key})
    elif user_message.lower().strip() == "":
        pass
    else:
        bot_response = bot.get_response(user_message)
        st.session_state.history.append({"message": user_message, "is_user": True, 'key': widget_1_key})
        st.session_state.history.append({"message": bot_response, "is_user": False, 'key': widget_2_key})


def main():
    init_bot_engine()

    if "history" not in st.session_state:
        st.session_state.history = []

    st.header("Your Personal Covid-19 Chatbot  :robot_face: :heartpulse:")
    st.subheader(" ")

    st.markdown("##### This is a Covid-19 Chatbot that can help you to know about the Covid-19.")

    st.info("Head to https://www.cowin.gov.in/ and Register yourself to get your vaccination dose :syringe:")

    st.markdown(HELP_TEXT)

    placeholder =  st.empty()
    st.text_input("Ask me something!", key="input_text", on_change=generate_response)

    with placeholder.container():
        for chat in st.session_state.history: 
            st_message(**chat)


main()