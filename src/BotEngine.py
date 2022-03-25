import os

from sympy import false
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import List

import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import re
import logging


logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("debug.log"),
                logging.StreamHandler()
            ],
            datefmt='%d-%b-%y %H:%M:%S'
        )

class BotEngine(object):
    '''
    A class to initialize the model and get the functionalities to for building a chatbot 
    given some pre-defined knowledged-base(answers and context)

    '''
    def __init__(
        self,
        model_path: str,
        data_path: str,
        help_text: str = '__help_text__',
        bot_prompt: str = 'BOT> ',
        user_prompt: str = 'YOU> ',
        similarity_threshold: float = 0.1,
    ) -> None:
        '''
        Initialize the BotEngine class
        args:
            model_path:str, path to the tf SavedModel folder
            data_path:str, path to the CSV/excel file containing the answers and context
            help_text:str, text to be displayed when the user enters 'help'
            bot_prompt:str, prompt for the bot
            user_prompt:str, prompt for the user
            similarity_threshold:float, threshold for the similarity between the question encodings and the answer encodings
        '''

        self.model_path = model_path
        self.data_path = data_path
        self.bot_prompt = bot_prompt
        self.user_prompt = user_prompt
        self.response_encodings = None
        self.help_text = help_text
        self.similarity_threshold = similarity_threshold
        self._default_response = 'Sorry, I don\'t understand. Please try again.'    
        
        assert os.path.exists(self.data_path), 'The data file does not exist'
        assert os.path.exists(self.model_path), 'The model file does not exist'
        assert os.path.exists(self.model_path + '/saved_model.pb'), 'The model file does not exist'
        assert self.similarity_threshold > -1.0 and self.similarity_threshold < 1.0, 'The similarity threshold should be between -1.0 and 1.0'
        
        # Read the file containing the answers and context. The file should be in CSV/Excel format
        self.data = self.__read_data(self.data_path)

        # Load the model
        self.model = self.__get_model(self.model_path)

    
    def __read_data(self, data_path: str)-> pd.DataFrame:
        '''
        Reads the CSV/Excel file containing answer and context and returns the dataframe
        args:
            data_path:str, path to the csv/excel file
        returns:
            dataframe:pd.DataFrame, dataframe containing the data(answers and context)
        '''
        logging.info('Reading the data file...')
        pd.set_option('max_colwidth', 100)

        if data_path.endswith('csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('xlsx') or data_path.endswith('xls'):
            data = pd.read_excel(data_path)
        else:
            logging.error('The data file is not in CSV/Excel format')
            raise ValueError('The data file should be in CSV/Excel format')
        logging.info('Data file read successfully!')
        return data
    
    def __get_model(self, model_path: str):
        '''
        Loads the as tf SavedModel from the path
        args:
            model_path:str, path to the model
        returns:
            model:tf.saved_model, loaded model
        '''
        logging.info('Loading the model...')
        try:
            model = tf.saved_model.load(model_path)
            logging.info('Model loaded successfully!')
        except Exception as e:
            logging.error('Error loading the model - ' + str(e))
            raise e
        return model

    def __preprocess_sentences(self, input_sentences: List[str])-> List[str]:
        '''
        Preprocesses the input sentences to be compatible with the model.(The model has never seen covid-19 or covid before, so we replace it with coronavirus)
        args:
            input_sentences:list, list of sentences
        returns:
            preprocessed_sentences:list, list of preprocessed sentences
        '''
        return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I) 
                for input_sentence in input_sentences]
    
    def __generate_response_encodings(self, model, ans_text:List[str], context_text:List[str])-> tf.Tensor:
        '''
        The response_encoder signature is used to encode the answer test and context test. The output is a 512 dimensional vector.
        args:
            model:tf.saved_model, loaded model
            ans_text:list, list of answer text
            context_text:list, list of context text
        returns:
            response_encodings:tf.Tensor, response encodings
        '''
        if isinstance(ans_text, str):
            ans_text = [ans_text]
        if isinstance(context_text, str):
            context_text = [context_text]
        
        assert len(ans_text) == len(context_text), logging.error('The number of answers and context should be equal')

        # Create response embeddings
        processed_ans_text = self.__preprocess_sentences(ans_text)
        processed_context_text = self.__preprocess_sentences(context_text)

        logging.info('Generating response encodings...')
        try:
            response_encodings = model.signatures['response_encoder'](
                    input=tf.constant(processed_ans_text),
                    context=tf.constant(processed_context_text))['outputs']
            logging.info('Response encodings generated!')
        except Exception as e:
            logging.error('Error generating response encodings - ' + str(e))
            raise e
        
        self.response_encodings = response_encodings
    
    def __get_question_encodings(self, model, question_text:List[str])-> tf.Tensor:
        '''
        The 'question_encoder' signature is used to encode variable length question test and the output is a 512 dimensional vector.
        args:
            model:tf.saved_model, loaded model
            question_text:list, list of question text
        returns:
            question_encodings:tf.Tensor, question encodings
        '''
        if isinstance(question_text, str):
            question_text = [question_text]

        # Create question embeddings
        logging.info('Generating question encodings...')
        
        try:
            processed_question = self.__preprocess_sentences(question_text)
            question_encodings = model.signatures['question_encoder'](
                    input=tf.constant(processed_question))['outputs']
            logging.info('Question encodings generated!')
        except Exception as e:
            logging.error('Error generating question encodings - ' + str(e))
            raise e
        
        return question_encodings
    
    def get_response(self, question:str)-> str:
        '''
        Get the response for the question
        args:
            question:str, question for which the response is to be generated
        returns:
            response:str, response for the question
        '''
        # Get the response encodings
        if self.response_encodings is None:
            self.__generate_response_encodings(self.model, self.data['Answer'], self.data['Context'])

        question_encodings = self.__get_question_encodings(self.model, question)
        
        # Get the cosine similarity between the response encodings and question encodings
        cosine_similarity = tf.reduce_sum(tf.multiply(self.response_encodings, question_encodings), axis=1)
        
        # Get the index of the response with the maximum cosine similarity
        max_cosine_similarity_index = tf.argmax(cosine_similarity).numpy()
        
        logging.debug(
            f'Cosing similarity: {cosine_similarity.numpy()} \n'
            f'Max cosine similarity: {tf.math.reduce_max(cosine_similarity)} \n'
            f'Max cosine similarity index: {max_cosine_similarity_index} \n'
        )
        # Return the response
        if tf.math.reduce_max(cosine_similarity) > self.similarity_threshold:
            return self.data.iloc[max_cosine_similarity_index]['Answer']
        else:
            return f'{self._default_response}\n {self.help_text}'

    def init_bot(self):
        while True:
            question = str(input(self.user_prompt))
            if question.strip().lower() == 'exit':
                break
            elif question.strip().lower() == 'help':
                print(self.help_text)
            elif question.strip().lower() == '':
                continue
            else:
                response = self.get_response(question)
                print(f'{self.bot_prompt} {response}')
