from app.BotEngine import BotEngine
import re
from typing import List


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

__help_text = '''
    Commonly asked questions:
        1. What is a coronavirus?
        2. What are the symptoms of COVID-19?
        3. How does COVID-19 spread?
        4. What can I do to protect myself from this virus?
        5. How long does the virus survive on surfaces?
        6. Vaccines for COVID-19?
    '''

def main():

    model_path = 'model/universal-sentence-encoder-multilingual-qa_3'
    data_path = 'WHO_FAQ.xlsx'

    covid_bot = BotEngine(
        model_path=model_path,
        data_path=data_path,
        preprocess_func=preprocess_sentences,
        help_text=__help_text,
    )

    covid_bot.init_bot()

if __name__ == "__main__":
    main()