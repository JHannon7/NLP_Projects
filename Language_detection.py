# THIS SCRIPT USES THREE METHODS FOR DETECTING THE LANGUAGE OF A PIECE OF TEXT

# LANGUAGE DETECTION WITH SPACY
# Do below if you haven't done it before
#pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz

#import modules
#import en_core_web_sm
#from spacy.language import Language # Must be spaCy v3
#from spacy_langdetect import LanguageDetector

from langdetect import detect, DetectorFactory
import langid
DetectorFactory.seed = 0

#SPACY LANGUAGE DETECTION
#def language_detector(input_text):
  #nlp = en_core_web_sm.load(disable=["tagger", "ner", "lemmatizer"])
  #nlp.add_pipe('language_detector', last=True)
  #doc = nlp(input_text)
  #language = doc._.language
 # return language
#print("Detected language:", str(language_detector(text)))


# LANGDETECT LANGUAGE DETECTION
def language_detector2(input_text):
    detection=detect(input_text)
    return detection

# LANGID LANGUAGE DETECTION
def language_detector3(input_text):
    detection=langid.classify(input_text)
    return detection


