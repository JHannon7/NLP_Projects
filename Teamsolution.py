# THIS SCRIPT FINDS KEY TERMS AND SUMMARIES OF PASSAGES RELATING TO WATER UTILISATION  #


# Author: Joseph Hannon


#import general modules
import math
import re
import json
import requests
from string import punctuation

#import NLP modules
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest

# upload stopwords and punctuation
stopwords = list(STOP_WORDS)
from nltk.corpus import stopwords
punctuation = punctuation + '\n'


#import pdf document extraction module
import fitz


# terms (naming conventions) used in water utiliities
terms = ["suppl","demand","flow","valve","pressure","customers","data"]

# variables for text extraction 

text = ""
url=""


# if document is provided by a url link, download and extract
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

# extract text from downloaded pdf file
if(url == ""):
     doc = fitz.open('TeamSolve.pdf')
     pdftext = []
     for page in doc:
         t = page.get_text().encode("ascii", "ignore").decode()
         pdftext.append(t)
     doc.close()
     str1 = ''.join(str(e) for e in pdftext)
     text = text+str1


# remove spaces between lines

text = " ".join(text.split())

# remove all line breaks from text 

text = text.replace('\\n', ' ')


# Get just one example paragraph

pos1=text.find(" 1. ")
pos2=text.find(" 2. ")
if(pos1>1 and pos2>1):
    text=text[pos1:pos2]
    

# Get the number of words

word_list = text.split()
number_of_words = len(word_list)

# calculate the summarisation percentage

sum_percent = round(((math.log10(number_of_words) * 40) / (number_of_words)),4)


# Tokenize the words from the sentences in text.

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)


#Create stop word list
stop_words = list(stopwords.words('english')) + list(STOP_WORDS)
stopwords=list(set(stop_words))


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


# Get the title (first sentence) and the named section

if len(str(sentence_tokens[0]))<5:
 subtitle=str(sentence_tokens[1])
 named_section=str(sentence_tokens[1])
else:
  subtitle=str(sentence_tokens[0])
  named_section=str(sentence_tokens[0])

# Remove stop words from title

stopwords2 = [' ' + x + ' ' for x in stopwords]
for x in stopwords2:
  subtitle=subtitle.replace(x,' ')
  
# Remove punctuation
subtitle = re.sub(r'[^\w\s]', '', subtitle)

# Put in quotes

subtitle = f'"{subtitle}"' 

# Get the non-named sections

demand_section=""
control_section=""
pressure_section=""
customers_section=""
data_section=""


for i in range(len(sentence_tokens)):
    if str(sentence_tokens[i]).lower().find(terms[0])>-1 or str(sentence_tokens[i]).lower().find(terms[1])>-1:
        demand_section+=str(sentence_tokens[i])
    if (str(sentence_tokens[i]).lower().find(terms[2])>-1 or str(sentence_tokens[i]).lower().find(terms[3])>-1) and str(sentence_tokens[i]).lower().find(terms[6])==-1:
        control_section+=str(sentence_tokens[i])
    if str(sentence_tokens[i]).lower().find(terms[4])>-1 and str(sentence_tokens[i]).lower().find(terms[6])==-1:
        pressure_section+=str(sentence_tokens[i])
    if str(sentence_tokens[i]).lower().find(terms[5])>-1:
        customers_section+=str(sentence_tokens[i])
    if str(sentence_tokens[i]).lower().find(terms[6])>-1:
        data_section+=str(sentence_tokens[i])
         

# Now get the entities #

entities=[]

for i in range(len(text)-3):
 if text[i]=="P" and text[i+1].isnumeric() and text[i+3].isalnum() == False:
  entity= f'"{text[i:i+3]+", pressure sensor"}"'
  entities.append(entity)
 if text[i]=="S" and text[i+1].isnumeric() and text[i+2].isalnum() == False:
  entity= f'"{text[i:i+2]+", pumping station"}"'
  entities.append(entity)
 if text[i]=="V" and text[i+1].isnumeric() and text[i+3].isalnum() == False:
  entity= f'"{text[i:i+3]+", valves"}"'
  entities.append(entity)#
 if text[i]=="C" and text[i+1].isnumeric() and text[i+3].isalnum() == False:
  entity= f'"{text[i:i+3]+", customer"}"'
  entities.append(entity)
 if text[i]=="F" and text[i+1]=="M" and text[i+2].isnumeric() and text[i+3].isalnum() == False:
  entity= f'"{text[i:i+3]+", flow meter"}"'
  entities.append(entity)

pos=text.find("DMA")
if pos>-1  and text[pos+4].isnumeric():
  entity= f'"{text[pos:pos+5]}"'
  entities.append(entity)
    
# remove duplicates    
entities =list(set(entities))
#sort in alphabetical order
entities.sort()


# Find subject-object relationships

relationships=[]

for i in range(len(sentence_tokens)):
 doc = nlp(str(sentence_tokens[i]))
 triplet=""
 count=0
 for nc in doc.noun_chunks:
    if(len(nc.text)>2) and count<3:
     triplet+=nc.text+","
     count=count+1
 relationships.append(triplet)
 
# remove any empty elements  
relationships = list(filter(None, relationships))
    
# Remove stop words 

for i in range(len(relationships)):
 for x in stopwords2:
  relationships[i]=relationships[i].replace(x,' ')
 relationships[i]=f'"{relationships[i]}"'


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


# Print out all features

print("\n")
print("TITLE:",subtitle,"\n")
print("SUMMARY:",summary,"\n")
print("NAMED SECTION: ",f'"{named_section}"'+"\n")
print("DEMAND SECTION: ",f'"{demand_section}"'+"\n")
print("CONTROL SECTION: ",f'"{control_section}"'+"\n")      
print("PRESSURE SECTION: ",f'"{pressure_section}"'+"\n")  
print("CUSTOMER SECTION: ",f'"{customers_section}"'+"\n") 
print("DATA SECTION: ",f'"{data_section}"'+"\n")
print("ENTITIES","\n")
print(entities,"\n")
print("RELATIONSHIPS","\n")
print(relationships,"\n")

 # Create a Python object:
     
obj = {
 "title:": subtitle,
 "summary": summary,
 "named section": named_section,
 "demand section": demand_section,
 "control_section": control_section,
 "pressure_section": pressure_section,
 "customer_section": customers_section,
 "data_section": data_section,
 "Entities=": entities,
 "Relationships": relationships
 }
 
 # convert into JSON:
jstring = json.dumps(obj)
 
#output json string to file
with open('TS_output.json', "w", encoding="utf-8") as file1:
   file1.write(jstring)
file1.close()

