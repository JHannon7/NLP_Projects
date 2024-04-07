    ## THIS SCRIPT PERFORMS NLTK AND SPACY NAMED ENTITY RECONGITION ###


import nltk
import spacy
from operator import itemgetter

# download nltk words
#nltk.download('words')
# load spacy model
nlp = spacy.load('en_core_web_sm')

def get_named_entities(text):
    
 #spacy analysis
 doc = nlp(text)
 entities=[]
 # print entities
 for ent in doc.ents:
   if ent.label_!="ORDINAL" and ent.label_!="CARDINAL" and ent.label_!="DATE" and ent.label_!="PERCENT" and ent.label_!="TIME" and ent.label_!="QUANTITY":
        entities.append(ent.text+": "+ent.label_)
     
 #nltk analysis
 entity_list=[]
 for sent in nltk.sent_tokenize(text):
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
        if hasattr(chunk, 'label'):
          entity_value=' '.join(c[0] for c in chunk)+(": ")+ chunk.label()
          entity_list.append(entity_value)   
 entity_list = [x.replace('ORGANIZATION', 'ORG') for x in entity_list]

 # Merge the lists
 entities=entities+entity_list

 # dedupe the entities in toto
 entities=list(set(entities))
 
 #Remove any entities containing urls
 
 entities=[x for x in entities if "http" not in x]

 # get the name of the entity alone
 named_entity = [ x[:x.find(": ")] for x in entities]
 
 #dedupe by name of entity alone
 
 for i in range(0,len(named_entity)):
      for j in range(i+1,len(named_entity)):
        if named_entity[i] == named_entity[j]:
          named_entity[j]=""
          entities[j]=""
 entities = list(filter(None, entities))
  
 # Now rank the entity names in terms of their frequency in the text 

 entity_count=[]
 for i in range(len(named_entity)):
   counter=0
   counter+=text.count(str(named_entity[i]))
   entity_count.append(counter)

 # sort function (descending) for all entities
 entity_count, entities = (list(x) for x in zip(*sorted(zip(entity_count, entities), reverse=True,key=itemgetter(0))))

 # limit number of named entities
 if len(entities)>=20:
  del entities[20:]

 return entities
