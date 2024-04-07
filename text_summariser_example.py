# TEXT SUMMARISER TOOL FOR PDFS/DOCS WITH KEYWORD EXTRACTION AND TOPIC MODELLING #


#import general modules
import math
import json
import re
import requests
from string import punctuation


#import randomisation modules/statistical functions
import random as ran
from random import seed
from random import random
from statistics import mode

#import NLP
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from heapq import nlargest
from rake_nltk import Rake
import yake
import gensim
from gensim import corpora
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# upload stopwords and punctuation
stopwords = list(STOP_WORDS)
punctuation = punctuation + '\n'

#import document extraction
import fitz
import docx

#import Language detection and full length translation
from Language_detection import language_detector2
from Language_detection import language_detector3
from sentiment_analysis import sentiment_analysis_textblob
from named_analysis import get_named_entities
import Full_translator as FT


# seed random number generator
seed(1)
text = ""


# example url
url="https://www.nato.int/nato_static_fl2014/assets/pdf/2022/6/pdf/290622-strategic-concept.pdf"

# download and extract data from url link to pdf file


if url.find('.pdf') != -1:
    response = requests.get(url, headers={'User-Agent': 'Mozilla'})
    with open('upload.pdf', 'wb') as f:
        f.write(response.content)
    doc = fitz.open('upload.pdf')
    pdftext = []
    for page in doc:
        t = page.get_text().encode("ascii", "ignore").decode()
        pdftext.append(t)
    doc.close()
    str1 = ''.join(str(e) for e in pdftext)
    text = text+str1

# extract text from pdf or docx file uploaded

try:

 if(url == ""):
     doc = fitz.open('upload.pdf')
     pdftext = []
     for page in doc:
         t = page.get_text().encode("ascii", "ignore").decode()
         pdftext.append(t)
     doc.close()
     str1 = ''.join(str(e) for e in pdftext)
     text = text+str1

except:
    
 if(url == ""):
    doc = docx.Document('upload.docx')
    fullText = []
    for para in doc.paragraphs:
       fullText.append(para.text)
    text = '\n'.join(fullText)
    

# remove spaces between lines

text = " ".join(text.split())

# remove all line breaks from text #

text = text.replace('\\n', ' ')

# Get the number of words

word_list = text.split()
number_of_words = len(word_list)

# calculate the summarisation percentage

sum_percent = round(((math.log10(number_of_words) * 20) / (number_of_words)),4)


# translate the article if it is not in English

if str(language_detector2(text)).find("en")==-1 and str(language_detector3(text)).find("en")==-1:
 text=FT.translate(text)

#Get the sub-title (first four words)

subtitle=text.split()[:4]
subtitle = ' '.join(subtitle)
subtitle = f'"{subtitle}"'

# Tokenize the words from the sentences in text.

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)


# Calculate word frequencies from the text after removing stopwords and punctuations.

word_frequencies = {}
for word in doc:
 if word.text.lower() not in stopwords:
  if word.text.lower() not in punctuation:
   if word.text not in word_frequencies.keys():
     word_frequencies[word.text] = 1
   else:
     word_frequencies[word.text] += 1


# Calculate the maximum frequency and divide it by all frequencies to get normalized word frequencies.

max_frequency = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency


# Get sentence tokens

sentence_tokens = [sent for sent in doc.sents]


# Calculate the most important sentences by adding the word frequencies in each sentence.

# Get sentence_scores
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]


# From headhq import nlargest and calculate percent of text with maximum score.

select_length = int(len(sentence_tokens)*sum_percent)
summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)


# final_summary
final_summary = [word.text for word in summary]
summary = ''.join(final_summary)

# add space after full stops and other punctuations marks

summary = re.sub(r'(?<=[.,])(?=[^\s])', r' ', summary)

# add quotes
summary = f'"{summary}"'


##################################
# sentiment analysis
##################################

        
sentiment=sentiment_analysis_textblob(text)
sentiment= f'"{sentiment}"'
        

##################################
#  Unique keyword extraction
##################################

# Get keywords using yake #

key_words=[]
max_ngram_size = 2
numOfKeywords = 20

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
        if x.lower() not in stopwords:
              new_words.append(x)

# take the top twenty          
new_words = new_words[0:20]


# merge with keywords
key_words=key_words+new_words

# remove punctuations and white spaces
for i in range(len(key_words)):
 key_words[i] = re.sub(r'[.,"\'?():!;]', '', key_words[i]).strip()
 key_words[i]=WordNetLemmatizer().lemmatize(key_words[i])
 

#remove duplicates among key words
key_words = list(set(key_words))

#remove individual numbers

for i in range(len(key_words)):
    if key_words[i].isnumeric()==True:
      key_words[i]=""
      
# remove empty elements in the list
key_words = [x for x in key_words if x != '']

# re-order alpahbetically
key_words.sort()


##################################
# KEY PHRASES (Top 10)
##################################

r=Rake()
r.extract_keywords_from_text(text)
key_phrases=r.get_ranked_phrases()[0:10]


##################################
# NAMED ENTITIES
##################################

named_entities=get_named_entities(text)


##################################
# TOPIC MODELING
##################################


# instantiate the Lemmatizer to perform stemming/lemmatization

lmr = WordNetLemmatizer()

# tokenize the text

article_doc = []
for t in word_tokenize(text):
    if t.isalpha():
        t = lmr.lemmatize(t.lower())
        if t.lower() not in stopwords:
            article_doc.append(t)
            
#create holding matrix
matrix = [[]]
matrix[0]=article_doc

#create a dictionary
mapping = corpora.Dictionary(matrix)
data = [mapping.doc2bow(word) for word in matrix]

#create lda model
ldamodel = gensim.models.ldamodel.LdaModel(data, num_topics = 10, id2word=mapping, random_state=0, eval_every=1,  passes=1)
topics = ldamodel.print_topics(num_words=5)


#create another holding matrix for the keywords for word2vec
topic_words=[]
matrix2 = [[]]
matrix2[0]=key_words


#remove punctuation from topic output
for topic in topics:
 topic_word = ','.join(map(str, topic))
 topic_word = re.findall('"([^"]*)"', topic_word)
 topic_word=','.join(topic_word)
 topic_words.append(topic_word)
 


# Initialise word2vec model
w2v_model = Word2Vec(matrix2, window=10, min_count=1,seed=20,workers=10)


topic_names=[]
words2=[]
sim_word=''


# Get topic names for all topics (from the list of keywords)
for i in range(len(topic_words)):    
 words = topic_words[i].split(",")
 words.sort()
 words2.append(words)
 #print(words)
 temp=[]
 for j in range(len(words)):  
  try:
   sim_words = w2v_model.wv.most_similar(words[j])
   sim_word = str(sim_words[ran.randint(0, len(sim_words)-1)])
   pos=sim_word.find(",")
   if(pos>-1):
    sim_word=sim_word[0:pos]
    sim_word = re.sub("[^a-zA-Z1-9 :\.\-]", "", sim_word)
    temp.append(sim_word)
   sim_word=mode(temp).upper() 
  except KeyError:
     sim_word=str(topic_words)
     sim_word=sim_word.split(',')[0].upper()
     sim_word = re.sub("[^a-zA-Z1-9 :\.\-]", "", sim_word)
 topic_names.append(str(sim_word))


# add topic labels to topic words
counter=0
for i in range(len(topic_words)):
 counter=counter+1
 topic_name ="Topic "+str(counter)+":"
 topic_word=topic_name,topic_names[i],topic_words[i]
 topic_words[i]=topic_word
 
 
# dedupe the topics if very similar
for i in range(0,len(words2)):
  for j in range(i+1,len(words2)):
    if words2[i]==words2[j]:
        topic_words[j]=""
topic_words = list(filter(None, topic_words))
    


# Print out all features

print("\n")
print("sub-title:",subtitle,"\n")
print("number of words:", number_of_words,"\n")
print("summary:",summary,"\n")
print("sentiment level:",sentiment,"\n")
print("Unique keywords:",key_words,"\n")
print("Unique keyphrases:",key_phrases,"\n")
print("Named entities:","\n")
for i in range(10):
 print(named_entities[i])
print("\n")
print("Topics:","\n")
for topic_word in topic_words:
 print(topic_word)
 
 
 # Put into JSON format 
 
#Python object:
    
obj = {
"sub-title:": subtitle,
"number_of_words": number_of_words,
"summary": summary,
"sentiment": sentiment,
"Unique keywords": key_words,
"Unique keyphrases": key_phrases,
"Named entities": named_entities,
"Topics=": topic_words,
}

# convert into JSON:
jstring = json.dumps(obj)

# output to a text file
with open("text_summary_output.json", "w") as json_file:
    json_file.write(jstring)