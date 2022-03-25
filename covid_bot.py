from app.BotEngine import BotEngine

__help_text = '''
    Commonly asked questions:
        1. What is a coronavirus?
        2. What are the symptoms of COVID-19?
        3. How does COVID-19 spread?
        4. How can I protect myself from this virus?
        5. How long does the virus survive on surfaces?
        6. Vaccines for COVID-19?
    '''

covid_bot = BotEngine(
    model_path='model/universal-sentence-encoder-multilingual-qa_3',
    data_path='WHO_FAQ.xlsx',
    help_text=__help_text
)

covid_bot.init_bot()