   # THIS SCRIPT PROVIDES FUNCTIONS TO TRANSLATE FULL LENGTH TEXT FROM ENGLISH TO A TARGET LANGUAGE #


#import modules
from deep_translator import GoogleTranslator
from nltk import sent_tokenize
from collections.abc import Iterable


def chunks(list_to_use, chunk_size):

    for i in range(0, len(list_to_use), chunk_size):

        yield list_to_use[i:i+chunk_size]


def flatten(multiD_list):

    for element in multiD_list:

        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)

        else:
            yield element


#define functions for translating from english to the target language

def google_translator(input_text):
 translation=GoogleTranslator(source='auto', target='en').translate(input_text)
 return translation


def break_up(input_text, max_characters, min_joins = 1):

    sent_text = sent_tokenize(input_text)

    while True:
        pairs_list = list(chunks(sent_text, 2))

        joins = 0

        for i in range(len(pairs_list)):
            if len(" ".join(pairs_list[i])) < max_characters:
                pairs_list[i] = " ".join(pairs_list[i])
                sent_text = list(flatten(pairs_list))
                joins = joins + 1
        
        if joins <= min_joins:
            break
    
    return sent_text

#translate working within operational limit set by Google
def translate(input_text, max_characters=4000):
    split_text = break_up(input_text, max_characters)

    translated_text = ""

    for chunk in split_text:
        translated_chunk = google_translator(chunk)
        translated_text = translated_text + " " + translated_chunk
    
    return translated_text



  