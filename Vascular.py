# import pdf text extraction
import PyPDF2
import pysummarization
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

#import general modules
import math
import re
import csv
import os
from string import punctuation

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor



#import randomisation modules/statistical functions

import random
random.seed(10)
#print(random.random())


#import NLP
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from heapq import nlargest
from rake_nltk import Rake
import yake


#import Language detection and full length translation
from named_analysis import get_named_entities


count=0
text = ''

# FOR LOOP #

pdf_dir = 'C:/Users/joseph.hannon/OneDrive - NEC Software Solutions/Desktop/Python/NVR'

for filename in os.listdir(pdf_dir):
  if filename.endswith('.pdf'):
      
   pdf_file = open(os.path.join(pdf_dir, filename), 'rb')
  
   pdf_reader = PyPDF2.PdfReader(pdf_file)

   for i in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    text += page.extract_text()
 
   pdf_reader = PyPDF2.PdfReader(pdf_file)

   for i in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    text += page.extract_text()

    
# remove spaces between lines
    
text = " ".join(text.split())
    
# remove all line breaks from text #
    
text = text.replace('\\n', ' ')

#remove all "yes"

text = text.replace('Yes',' ')
text = text.replace('No',' ')

   
#print(text)
    
# Get the number of words

word_list = text.split()
number_of_words = len(word_list)


# Tokenize the words from the sentences in text.

#nlp = spacy.load('en_core_web_sm')

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)


##################################
#  Get Summary
##################################

# Object of automatic summarization.
auto_abstractor = AutoAbstractor()
# Set tokenizer.
auto_abstractor.tokenizable_doc = SimpleTokenizer()
# Set delimiter for making a list of sentence.
auto_abstractor.delimiter_list = ['.', '\n']
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Summarize document.
result_dict = auto_abstractor.summarize(text, abstractable_doc)


summary=""

for sentence in result_dict['summarize_result']:
    summary+=sentence


#summary=summarize(text, word_count=250, ratio=0.5)

          
print (summary)
#print (keywords(text))

##################################
#  Unique keyword extraction
##################################

# Get keywords using yake #

key_words=[]
max_ngram_size = 3
numOfKeywords = 75

kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords(text)

custom_kw_extractor = yake.KeywordExtractor( n=max_ngram_size, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)


for kw in keywords:
    keyline=str(kw)
    res = ''.join([i for i in keyline if i.isalpha() or i.isspace()])
    key_words.append(res)
  

##################################
# most frequent words in the text
##################################


# count the words in the text

cnt = Counter()
for word in text.split(' '):
 cnt[word] += 1
 
total= ','.join(map(str, cnt))

#make into list
extra_words = total.split(',')
extra_words.sort()

# remove stop words from words made lower case

new_words=[]
for x in extra_words:
        if x not in stopwords.words('english'):
              new_words.append(x)

# take the top 50          
new_words = new_words[0:75]


# merge with keywords
key_words=key_words+new_words

# remove punctuations and white spaces
for i in range(len(key_words)):
 key_words[i] = re.sub(r'[.,"\'?():!;]', '', key_words[i]).strip()
 key_words[i]=WordNetLemmatizer().lemmatize(key_words[i])
 

#remove duplicates among key words
key_words = list(set(key_words))

#remove individual numbers and keywords excessively long

for i in range(len(key_words)):
    if key_words[i].isnumeric()==True:
      key_words[i]=""
    if len(key_words[i])>25:
      #print(keywords[i])
      key_words[i]=""
      
# remove empty elements in the list
key_words = [x for x in key_words if x != '']

# re-order alpahbetically
key_words.sort()


for i in range(len(key_words)):
 key_words[i]=key_words[i].replace(" e","")
 if key_words[i][0].isdigit():
    key_words[i]=""

key_words=list(filter(None,key_words))

##################################
# KEY PHRASES (Top 10)
##################################

r=Rake()
r.extract_keywords_from_text(text)
key_phrases=r.get_ranked_phrases()[0:15]
key_phrases=list(set(key_phrases))

for i in range(len(key_phrases)):
    size=len(key_phrases[i])
    pos=key_phrases[i].find(".")
    if pos>-1 or size >100:
     key_phrases[i]=""

key_phrases=list(filter(None,key_phrases))

##################################
# NAMED ENTITIES
##################################

named_entities=get_named_entities(text)

for i in range(len(named_entities)):
    size=len(named_entities[i])
    if size<10:
     named_entities[i]=""
    for j in range(len(named_entities)):
        if i is not j:
         pos=named_entities[i].find(named_entities[j])
         if(pos>-1):
            named_entities[j]=""
named_entities=list(filter(None,named_entities))

##################################
# COMBINE # Now combine keywords and named entities
##################################


key_terms = key_words + named_entities
#key_words.sort()


# Get sentences relating to the keywords and entities

sentences=[]
for i in range(len(key_terms)):
  if text.find(key_terms[i]) != -1:
      
   k = random.randrange(len(text))
   
   pos=text.find(key_terms[i],k)
   
   if(pos==-1):
      pos=text.find(key_terms[i],0) 
     
   pos1=text.rfind(". ",0,pos)
   if pos1 != -1: 
    pos2=text.find(". ",pos+1)
   elif pos1 == -1:
       pos1= 0
       pos2=text.find(". ",pos+1)  
   sentence=text[pos1: pos2+1]
   sentence = re.sub(r'[^\w\s]','',sentence)
   sentence.lstrip()
   #sentence=f'"{sentence}"'
   sentences.append(sentence)
  else:
   sentences.append("")


# Print out all features

print("\n")
print("Summary: ",summary,"\n")
print("Unique keywords:",key_words,"\n")
print("Unique keyphrases:",key_phrases,"\n")
print("Named entities:",named_entities,"\n")



# Output to CSV

header = ['Term','Entities']
rows = zip(key_words,named_entities)
lines = zip(key_terms,sentences)


#with open('output.txt', 'w') as f:
 #   f.write(text)

        
with open('vascular_sentences.csv', 'w', encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for line in lines:
        writer.writerow(line)
        

 