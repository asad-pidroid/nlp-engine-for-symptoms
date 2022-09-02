from urllib import response
from django.http import HttpResponse
from django.shortcuts import render
from requests.auth import HTTPBasicAuth
from django.http import JsonResponse
import os
import requests
import json
import pandas as pd
import numpy as np
import re
import string, time
from textblob import TextBlob
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
#from hamcrest import none
#Semantic Textual Similarity
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
WordNetLemmatizer = WordNetLemmatizer()
from django.core.cache import cache
from rest_framework.views import APIView
from rest_framework.response import Response







# Download model comment after first run
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


#Removing html tags
def remove_html_tags(text):
    pattern = re.compile('[<,*?,>]')
    return pattern.sub(r'', text)


#Remove url tag
def remove_url(text):
    pattern = re.compile('http?//S+|www.S+')
    return pattern.sub(r'', text)


#Remove puncuation
exclude = string.punctuation
def remove_punct(text):
    for char in exclude:
        text = text.replace(char, '')
        return text.translate(str.maketrans(' ',' ',exclude))



#Spelling correction
def spell_correction(setense):
    incorrect_text = setense
    textblob = TextBlob(incorrect_text)
    correct_text = textblob.correct()
    return correct_text


#Stop word Remover
def stop_word_remover(sent):
    tokens = nltk.word_tokenize(sent)
    stop_words = stopwords.words('english')
    
    words_without_stop_words = []
    for word in tokens:
        if word in stop_words:
            continue
        else:
            words_without_stop_words.append(word)
    return words_without_stop_words



#test data
def temp_dr_transcription():
    trns = 'Dry COUGH 1.5 MONTHS CHEST PAIN ON COUGHING FEVER WT LOSS'
    trns = 'Patient is suffering from nausea and vomiting with chills and fever'
    trns = 'b/l cervical lymphadenopathy with fever & night sweat for 5 months'
    trns = 'routine immunization'
    trns = 'PAIN IN BACK AND UPPER ABDOMEN'
    trns = 'F/U/C OF TUBERCULOMA WITH MODERATE UNDERWEIGHT AND MODERATE WASTING'
    trns = 'LEFT SIDED NASAL OBSTRUCTION WITH THROAT DISCOMFORT'
    trns = 'no h/o fever palpitation breathing difficulty and orthopnea' ###
    trns = 'abdominal swelling'
    trns = 'swelling over right side of cheek for 1 year'
    trns = 'B/L DECREASED HEARING SENSITIVITY'
    trns = 'INCREASED FREQUENCY OF MICTURATION SINCE 20 DAYS'
    trns = 'WITH BURNING SENSATION IN ORAL CAVITY'
    trns = 'MULTIPLE REDDISH ITCHY LESION PRESENT OVER BREAST AND ABDOMEN*2 MONTHS'
    trns = 'A LARGE LOBULATED SOLID CYCTIC LESION IN THE MIDLINE &  AT THE INFRAUMBILICAL REGION INVOLVING THE ANTERIOR ABDOMINAL WALL AS DESCRIBED'
    trns = 'b/l pain in hypogastrium radiates towards chest from 2 months'
    trns = 'DOV FOR NEAR AND FAR L/E A/W OCULAR STRAINING AND FRONTOTEMPORAL HEADACHE x 1 MONTHx 1 MONTH'
    trns = 'ITCHING B/E AND WATERING B/E (SEASONAL) X 1 MONTH (AGGRAVATED)'
    trns = 'PAIN AND BLOOD IN URINE X 2 DAYS'
    return trns





#Preprocess the Text 
def clean_text(dr_transcription):
    #symptoms_plus_disease = 'right side abdominal hernias pain associated with increased after eating x 1 months and fever'
    #symptoms_plus_disease = 'complaints of itchy lesions over the legs and below the right buttock, hand, leg, on abdomen'
    #print(dr_transcription)
    symptoms_plus_disease = dr_transcription
    symptoms_plus_disease = symptoms_plus_disease.lower()
    symptoms_plus_disease = remove_html_tags(symptoms_plus_disease)
    symptoms_plus_disease = remove_url(symptoms_plus_disease)
    symptoms_plus_disease = remove_punct(symptoms_plus_disease)
    symptoms_plus_disease = stop_word_remover(str(symptoms_plus_disease))
    dr_transcription_processed = []
    for word in symptoms_plus_disease:
        lem_word = WordNetLemmatizer.lemmatize(word, pos='v')
        dr_transcription_processed.append(lem_word)
    #symptoms_plus_disease_wtokenize = word_tokenize(symptoms_plus_disease)
    #symptoms_plus_disease_stokenize = sent_tokenize(symptoms_plus_disease)
    
    #symptoms_plus_disease, 
    return dr_transcription_processed


#Full Body Redness
def findMg(dr_transcription):
    str1 = dr_transcription
    try:
        str1 = str1.lower()
        result = re.findall(r"\bfull body\b", str1)
        return result
    except Exception as e:
        return e


#Vitamin A/B?
def vitamin(dr_transcription):
    str1 = dr_transcription
    #print("in vitamin ",str1)
    #str = 'Vitamin b12 and Hypervitaminosis A'
    
    try:
        str1 = str1.lower()
        match = re.findall(r'\bvitamin [a-e] deficiency\b|\bvitamin \w antagonist\b|\bhypervitaminosis \w*\b|\bvitamin \w\d* deficiency\b|\bvitamin [a-e] deficiency\b|\bvitamin [a-e] \b|\bvitamin \w\d*\b|\bvitamin d dependent rickets\b|\bvitamin d dependent rickets type 1b\b', str1)
        return match
    except Exception as e:
        return e
    #return match   #or match[0]

#type 1 or type 2
def type1(dr_transcription):
    str1 = dr_transcription
    #str1 = str1.lower()
    try:
        str1 = str1.lower()
        match = re.findall(r'\btype \d diabetes\b|\bdiabetes mellitus type \d\b|\bglutaric aciduria type \d\b|\bprimary hyperoxaluria type \d\b|\bfull body \w*\b|\bfull body \w* \w*\b|\b\w* of the \w*\b|\b\w* of \w*\b|\bback lower \w*\b|\bback upper \w*\b|\bback pain\b|\blump in \w*\b|\blump on \w*\b|\bitching on \w*\b|\bchange in \w* \w*\b|\bchange in \w*\b|\bchanges in \w* \w*\b|\bchanges in \w*\b|\bswelling in \w*\b|\bswelling on \w*\b|\bvitamin d dependent rickets\b|\bpain in lower back\b|\bpain in mid back\b|\bbruise on \w*\b|\bpain in back\b', str1)
        return match
    except Exception as e:
        return e

    





#Special Keyword Conversion
def getLocalKeywordMeaning(w):
    spKeywordsData = json.load(open('static/specialKeywords/specialKeywords.json'))
    w = w.lower()
    if w in spKeywordsData:
        rt = spKeywordsData[w]
        return rt[0]
    else:
        return w

#Body json
def getBodyOrgan(w):
    bodyOrganData = json.load(open('static/specialKeywords/bodyOrgan.json'))
    w = w.lower()
    if w in bodyOrganData:
        rt = bodyOrganData[w]
        return rt
    else:
        return 0    
     
       
    

### Key Extraction for matching in prblemName for ngram=(1,2)
def key_words_extraction(dr_transcription):
    
    from keybert import KeyBERT
    #dr_trans = 'Paitent  suffering from nausea,  vomitting with chills Fever'
    #dr_trans = 'neck pain with headache since 6 months'
    #dr_trans = 'SENSITIVITY IN ALL TEETH and teeth pain'
    dr_trans = dr_transcription
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(dr_trans, keyphrase_ngram_range=(2, 2))
    #print(keywords)
    extracted_keyword = []
    try:
        for i in range(len(keywords)):
            #print("Keywords 2-gram :",keywords[i][0])
            #take keyword mtached to  special keywords  
            #keywords = keywords[i][0]
            nk = getLocalKeywordMeaning(keywords[i][0])
            #print(type(nk))
            extracted_keyword.append(nk)
            #extracted_keyword.append(keywords[i][0])
            #print("keyword in 2-gram",extracted_keyword)
        return extracted_keyword
    except Exception as e:
        return e




### Key Extraction for matching in prblemName for ngram=(1,1)
def key_words_extraction0(dr_transcription):
    from keybert import KeyBERT
    #dr_trans = 'Paitent  suffering from nausea,  vomitting with chills Fever'
    #dr_trans = 'neck pain with headache since 6 months'
    #dr_trans = 'SENSITIVITY IN ALL TEETH and teeth pain'
    dr_trans = dr_transcription
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(dr_trans, keyphrase_ngram_range=(0, 1))
    #print(keywords)
    extracted_keyword = []
    vitx = vitamin(dr_trans)
    #print("vitx return",vitx)
    if len(vitx) != 0:
        #print(vitx[0])
        extracted_keyword.append(vitx[0])

    try:
        for i in range(len(keywords)):
            #print("Keywords 0-gram",keywords[i][0])
            nk = getLocalKeywordMeaning(keywords[i][0])
            #print(type(nk))
            extracted_keyword.append(nk)
            #extracted_keyword.append(keywords[i][0])
            #print("keyword in 0-gram",extracted_keyword)
        return extracted_keyword

    except Exception as e:
        return e

### Key Extraction for matching in prblemName for ngram=(1,3)
def key_words_extraction3(dr_transcription):
    from keybert import KeyBERT
    #dr_trans = 'Paitent  suffering from nausea,  vomitting with chills Fever'
    #dr_trans = 'neck pain with headache since 6 months'
    #dr_trans = 'SENSITIVITY IN ALL TEETH and teeth pain'
    dr_trans = dr_transcription

    #print(dr_trans)
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(dr_trans, keyphrase_ngram_range=(2, 3))
    #print(keywords)
    extracted_keyword = []
    typx = type1(dr_trans)
    if len(typx) != 0:
        #print(typx)
        nk = getLocalKeywordMeaning(typx[0])
        extracted_keyword.append(nk)
    vitx = vitamin(dr_trans)
    #print("vitx return",vitx)
    if len(vitx) != 0:
        #print(vitx[0])
        nk = getLocalKeywordMeaning(vitx[0])
        extracted_keyword.append(nk)

    for i in range(len(keywords)):
        #print("Keywords 3-gram",keywords[i][0])
        nk = getLocalKeywordMeaning(keywords[i][0])
        #print(type(nk))
        extracted_keyword.append(nk)
        #extracted_keyword.append(keywords[i][0])
        #print("keyword in 3-gram",extracted_keyword)
    return extracted_keyword




#find day month week and years
#import re
def findDMW(dr_transcription):
    str1 = dr_transcription
    try:
        str1 = str1.lower()
        # search() for letter word surrounded by space
        # \b is used to specify word boundary
        result = re.findall(r"\b\d. day\b|\b\d week\b|\b\d*\.?\d+ month\b|\b\d+ days\b|\b\d+ weeks\b|\b\d+ week\b|\b\d*\.?\d+ months\b|\b\d+day\b|\b\d+days\b|\b\d+week\b|\b\d+weeks\b|\b\d+month\b|\b\d*\.?\d+months\b|\b\d+ year\b|\b\d+ years\b|\b\d+ yrs\b|\b\d+yrs\b|\b\d+ yr\b", str1)
        return result
    except Exception as e:
        return None


#find MG
def findMg(dr_transcription):
    str1 = dr_transcription
    str1 = str1.lower()
    result = re.findall(r"\b\d*\.?\d+ mg\b|\b\d*\.?\d+mg\b", str1)
    return result


#find acute severe serious
def findAcute(dr_transcription):
    str1 = dr_transcription
    str1 = str1.lower()
    result = re.findall(r"\bacute\b|\bsevere\b|\bserious\b|\bgrave\b|\bcritical\b", str1)
    return result


#find lower upper back
def findLUB(dr_transcription):
    str1 = dr_transcription
    str1 = str1.lower()
    result = re.findall(r"\blower\b|\bupper\b|\bback\b|\bmiddle\b|\bleft\b|\bright\b|\blow\b|\bover\b", str1)
    return result






#getting disease list
def get_disease_list():
    test_api =  'http://182.156.200.179:332/api/v1.0/Knowmed/getAllProblemList'
    #payload = {"alphabet": "a"}
    test_response = requests.post(test_api, json={"alphabet": ""})
    response_data = test_response.json()



    # Problem Names in list
    problemNames = []
    for i in range(len(response_data['responseValue'])):
        #print(response_data['responseValue'][i]['problemName'])
        problemNames.append(response_data['responseValue'][i]['problemName'].lower())
    #print(problemNames)

    disease_list = problemNames
    #objects = cache.set(disease_list)
    # disease_list = pd.read()
    # disease_list['Disease Name'] = disease_list['Disease Name']
    # disease_list['Disease Name'] = disease_list['Disease Name'].astype(str)
    # disease_list['Disease Name'] = disease_list['Disease Name'].apply(lambda x:x.lower())
    # disease_list = disease_list['Disease Name'].to_list()

    return disease_list


#key
def diseaseDetails(drTranscriptionDisease):
    test_api =  'http://182.156.200.179:332/api/v1.0/Knowmed/getAllProblemList'
    #payload = {"alphabet": "a"}
    test_response = requests.post(test_api, json={"alphabet": ""})
    response_data = test_response.json()
    #response_data = get_disease_list()
    #mydict = {}
    problemNames = {}
    for i in range(len(response_data['responseValue'])):
        #print(response_data['responseValue'][i]['problemName'])
        #problemNames.append(response_data['responseValue'][i]['id'])
        t = {response_data['responseValue'][i]['id']:response_data['responseValue'][i]['problemName'].lower()}
        problemNames.update(t)
        #x = [{id: id, 'problemName': problemName} for id, problemName in mydict.items() if problemName == search_value]
    drTranscriptionDisease= drTranscriptionDisease
    drx = []
    for dis in drTranscriptionDisease:
        [drx.append({'id':key,'problemNames':value}) for key, value in problemNames.items() if value == dis]
    return drx



#Checking disease
def check_disease(dr_transcription_processed):
    #Disease Matching

    diseases_list = get_disease_list()

    #embeddings = model.encode(dr_transcription_processed)
    dr_transcription_processed = dr_transcription_processed

    dr_transcription_processed2 = key_words_extraction(dr_transcription_processed)
    dr_transcription_processed0 = key_words_extraction0(dr_transcription_processed)
    dr_transcription_processed3 = key_words_extraction3(dr_transcription_processed)
    
    #diseases_list = ['fever', 'achondroplasia', 'abdominal hernias', 'pain', 'hernias', 'itchy', 'itch', 'cough', 'dry cough', 'chest pain','nausea', 'vomit']

    #diseases_list = diseases_list.append('headache')
    
    

    #print(dr_transcription_processed)

    # for sentence, embedding in zip(dr_transcription_processed, embeddings):
    #     print("Word:", sentence)
        #print("Embedding:", embedding)
    try:    
        matched_disease = []
        for i in dr_transcription_processed0:
            if i in diseases_list:
                matched_disease.append(i)
                continue
            # else:
            #     pass

        for i in dr_transcription_processed2:
            if i in diseases_list:
                matched_disease.append(i)
        
        for i in dr_transcription_processed3:
            if i in diseases_list:
                matched_disease.append(i)



        # if (len(matched_disease) <=3 ):
        #     #print("processed ngram 0")
        #     #dr_transcription_processed = dr_transcription_processed0
        #     print("processed ngram 0 drTrns", dr_transcription_processed0)
        #     for i in dr_transcription_processed2:
        #         if i in diseases_list:
        #             matched_disease.append(i)
        #         else:
        #             pass
        # elif len(matched_disease) == 0:
        #     print("processed ngram -3")
        #     #dr_transcription_processed = dr_transcription_processed3
        #     for i in dr_transcription_processed3:
        #         if i in diseases_list:
        #             matched_disease.append(i)
        #         else:
        #             pass
        # else:
        #     pass
    except Exception as e:
        return e


    #print("Matched Disease :->", matched_disease)

    #embedding_disease = model.encode(diseases_list)
    #print(embedding_disease)
    #sim = model.encode(dr_transcription)
    #print("comparing of 1st with 2nd :->","{0:.4f}".format(sim.tolist()[0][0]))
    matched_disease = list(set(matched_disease))
    return matched_disease


#get body organ list
def get_body_organ():
    # body_organ_list = pd.read_csv('body_organ.csv', encoding='cp1252')
    # body_organ_list['Organ'] = body_organ_list['Organ'].astype(str)
    # body_organ_list['Organ'] = body_organ_list['Organ'].apply(lambda x:x.lower())
    # body_organ_list = body_organ_list['Organ'].to_list()
    body_organ_list = ['abdomen', 'abdomen right hypochondium', 'abdomen (right lumber)', 'abdomen (right eliac region)', 'abdomen (left hypochondium)', 'abdomen (left lumber)', 'abdomen (left eliac region)', 'abdomen (epigastric region)', 'abdomen (umbilical region)', 'abdomen (hypogastrium)', 'ankle', 'back', 'back (lower)', 'back (upper)', 'breast', 'buttock', 'calf', 'chest', 'ear', 'elbow', 'forehead', 'eye', 'face', 'finger', 'foot', 'hair', 'hand', 'head', 'heel', 'hip', 'knee', 'leg', 'lips', 'mouth', 'nail', 'neck', 'nose', 'palm', 'pelvis', 'shin', 'shoulder', 'skin', 'teeth', 'thigh', 'throat', 'thumb', 'toe', 'waist', 'wrist', 'full body', 'anywhere', 'vagina', 'penis', 'nipple', 'heart', 'kidney', 'lungs', 'liver', 'brain', 'bladder', 'stomach', 'intestines', 'uterus', 'blood', 'overy', 'testicles', 'limbs', 'spine', 'gollbladder', 'skull', 'gums', 'parotid gland', 'upper extremities', 'lower extremities', 'not specified', 'salivary glands', 'cartilagenous', 'multisystemic disorders', 'neurological disorder', 'bone', 'blood vessel', 'pancreas', 'adrenal glanel', 'lymphatic system', 'larynx', 'genital', 'cervix', 'bile duct', 'cns', 'prostate', 'respiratory system', 'skeletal system', 'renal system', 'cardivascular system', 'pitutiary gland', 'pneal gland', 'hypohealauous', 'parathyroid gland', 'thyroid gland', 'gastrointestinal system', 'anus', 'anal anal', 'oesophagus', 'rectum', 'cavernous sinus', 'tongue', 'paranasal sinus', 'hardplate', 'softplate', 'male reproductive system', 'female reproductive system', 'fallopian tubes', 'lower abdomen', 'adrenal glands', 'joint', 'multiple organ', 'gallbladder', 'muscle', 'muscle skeletal system', 'tissues', 'immune system  ', 'urinary tract infection', 'spinal cord', 'bone marrow', 'utreine corpus', 'testis', 'oral cavity', 'oropharynx', 'red blood cells', 'ans', 'hypothalamus', 'arteries', 'sole', 'trunk', 'labia majora', 'labia minora', 'glans penis', 'mouth angle', 'buccal mucosa', 'shaft penis', 'haematopoietic system', 'trachea', 'circulatory system', 'peripheral nervous system', 'depends on area of burn', 'vascular system', 'venous system', 'urinary bladder', 'mammary glands', 'reproductive system', 'urinary system']
    return body_organ_list

#checking  body organ
def check_body_organ(dr_transcription):
    dr_transcription = dr_transcription
    #print("from body organ ", dr_transcription)
    #body_organ_list = get_body_organ()
    #body_organ_list = ['abdomen', 'abdomen (right hypochondium)', 'abdomen (right lumber)', 'abdomen (right eliac region)', 'abdomen (left hypochondium)', 'abdomen (left lumber)', 'abdomen (left eliac region)', 'abdomen (epigastric region)', 'abdomen (umbilical region)', 'abdomen (hypogastrium)', 'ankle', 'back', 'back (lower)', 'back (upper)', 'breast', 'buttock', 'calf', 'chest', 'ear', 'elbow', 'forehead', 'eye', 'face', 'finger', 'foot', 'hair', 'hand', 'head', 'heel', 'hip', 'knee', 'leg', 'lips', 'mouth', 'nail', 'neck', 'nose', 'palm', 'pelvis', 'shin', 'shoulder', 'skin', 'teeth', 'thigh', 'throat', 'thumb', 'toe', 'waist', 'wrist', 'full body', 'anywhere', 'vagina', 'penis', 'nipple', 'heart', 'kidney', 'lungs', 'liver', 'brain', 'bladder', 'stomach', 'intestines', 'uterus', 'blood', 'overy', 'testicles', 'limbs', 'spine', 'gollbladder', 'skull', 'gums', 'parotid gland', 'upper extremities', 'lower extremities', 'not specified', 'salivary glands', 'cartilagenous', 'multisystemic disorders', 'neurological disorder', 'bone', 'blood vessel', 'pancreas', 'adrenal glanel', 'lymphatic system', 'larynx', 'genital', 'cervix', 'bile duct', 'cns', 'prostate', 'respiratory system', 'skeletal system', 'renal system', 'cardivascular system', 'pitutiary gland', 'pneal gland', 'hypohealauous', 'parathyroid gland', 'thyroid gland', 'gastrointestinal system', 'anus', 'anal anal', 'oesophagus', 'rectum', 'cavernous sinus', 'tongue', 'paranasal sinus', 'hardplate', 'softplate', 'male reproductive system', 'female reproductive system', 'fallopian tubes', 'lower abdomen', 'adrenal glands', 'joint', 'multiple organ', 'gallbladder', 'muscle', 'muscle skeletal system', 'tissues', 'immune system  ', 'urinary tract infection', 'spinal cord', 'bone marrow', 'utreine corpus', 'testis', 'oral cavity', 'oropharynx', 'red blood cells', 'ans', 'hypothalamus', 'arteries', 'sole', 'trunk', 'labia majora', 'labia minora', 'glans penis', 'mouth angle', 'buccal mucosa', 'shaft penis', 'haematopoietic system', 'trachea', 'circulatory system', 'peripheral nervous system', 'depends on area of burn', 'vascular system', 'venous system', 'urinary bladder', 'mammary glands', 'reproductive system', 'urinary system']
    #print(body_organ_list)
    #body_organ_list = set(body_organ_list)
    # dr_transcription2 = key_words_extraction(dr_transcription)
    # print("bodycheck key 2",dr_transcription2)
    matched_body_organ = []
    #xbo = getBodyOrgan()
    # bo = getLocalKeywordMeaning(bo)
    #         print("bd spcl res",bo)
    for i in dr_transcription:
        bo = getBodyOrgan(i)
        #print("json res body organ", bo)
        if bo != 0:
            matched_body_organ.append(bo)
            
        # if i in body_organ_list:
        #     #print(i)
        #     matched_body_organ.append(i)
        #     continue


        # else:
        #     pass
            #print("none")


    #print("Organ :->", matched_body_organ)
    return matched_body_organ



#apiview
class DrTranscription(APIView):
    def post(self, request):
        dr_transcription= request.GET.get('transcription')




        # api_user = 'H!$$erV!Ce'
        # access_token = '0785C700-B96C-44DA-A3A7-AD76C58A9FBC'
        # #url = 'http://172.16.61.6:201/api//####'

        # #url_dr_transcription = 'http://172.16.61.6:201/api/#'
        
        # data = {"pid": 2154772}
        
        # payload = {"pid": 2154772, "DataMainID": 7039}

        # #r = requests.get(url,  headers={'Authorization':'Basic %s' % 'access_token'})

        # #response = requests.post(url_dr_transcription, data=payload, auth=HTTPBasicAuth('api_user', 'access_token',))
        # # test_response = requests.get(test_api)
        # # response_data = test_response.json()
        # #print(response_data)


        # test_api =  'http://182.156.200.179:332/api/v1.0/Knowmed/getAllProblemList'
        # # payload = {"alphabet": ""}
        # test_response = requests.post(test_api, json={"alphabet": ""})
        # response_data = test_response.json()
        # #print(response_data)

        # # problemName = response_data['responseValue'][0]
        # # problemId = response_data['responseValue'][0]
        # #print(problemId,problemName)
        #         #for temporary



        #dr_transcription = temp_dr_transcription()
        
        #print('response ',type(dr_transcription))
       
        
        dMW = findDMW(dr_transcription)
        mg = findMg(dr_transcription)
        condition = findAcute(dr_transcription)
        portion = findLUB(dr_transcription)
        
        disease = check_disease(dr_transcription)
        #print("Disease in call", disease)
        finalDiseaseList = diseaseDetails(disease)
        dr_transcription = clean_text(dr_transcription)
        bdOrgan = check_body_organ(dr_transcription)
        #print(bd_organ)
        bodyOrganList = get_body_organ()
        l1 = range(1,len(bodyOrganList))
        d = dict(zip(l1,bodyOrganList))
        matchedBodyOrgan = bdOrgan
        bodyOrgan = []
        for bo in matchedBodyOrgan:
                [bodyOrgan.append({'id':key,'bodyorgan':value}) for key, value in d.items() if value == bo]

        data = {
                
                'finalDiseaseList': finalDiseaseList,
                'bodyOrgan': bodyOrgan,
                'dMW' : dMW,
                'mg' : mg,
                'condition' : condition,
                'portion': portion,
            }
        #print("Body organ :", bd_organ)
        return Response( data, status=200)





#dr_transcription = get_dr_transcription(request=200)
#dr_transcription = temp_dr_transcription()
# before_clean_keyword = key_words_extraction(dr_transcription)
# dr_transcription = clean_text(dr_transcription)
# disease = check_disease(before_clean_keyword)
# bd_organ = check_body_organ(dr_transcription)
