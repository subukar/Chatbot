
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:34:44 2019

@author: Subasish
"""

import nltk
import numpy as np
import tflearn as tl
import tensorflow as tf
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()

with open("intent.json") as file:
    data= json.load(file)
#print(data["intents"])

try:
    
    with open("data.pickle","rb") as f:
        words,labels,training,output= pickle.load(f)

except:
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]
    # iterating through all the values of intent key in the data dictionary called intents.json
    for intent in data["intents"]:
        # iterating through all the values with pattern as key i.e all the questions.  
        for pattern in intent["patterns"]:
            #tokenizing(splitting by space) all the questions(patterns) and storing in variable called wrds
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        # creating a list of all the tags in the list called labels
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
    '''print(words)
    print(labels)
    print(docs_x)
    print(docs_y)'''
    
    words=[stemmer.stem(w.lower()) for w in words if w != "?"]
    
    words= sorted(list(set(words)))
    
    labels= sorted(labels)
    
    training=[]
    output=[]
    
    out_empty=[0 for x in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag=[]
        
        wrds=[stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row=out_empty[:]
        output_row[labels.index(docs_y[x])]=1
        
        training.append(bag)
        output.append(output_row)
    
    training=np.array(training)
    output=np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)

tf.reset_default_graph()
net= tl.input_data(shape=[None,len(training[0])])
net= tl.fully_connected(net,20)
net= tl.fully_connected(net,15)

net= tl.fully_connected(net,10)
net= tl.fully_connected(net,len(output[0]),activation="softmax")

net=tl.regression(net)

model2=tl.DNN(net)

try:
    model2.load("model2.tf")
except:
    model2.fit(training,output,n_epoch=10000,batch_size= 10, show_metric= True)
    model2.save("model2.tf")

def bag_of_words(s,words):
    bag=[0 for i in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
                
    return np.array(bag)

def chat():
    print("\nHey there!You can start talking now !!! or   (type quit to stop.)")
    while True:
        inp=input("You: ")
        if inp.lower()=="quit":
            break
        results= model2.predict([bag_of_words(inp,words)])[0]
        #print(results)
        results_index= np.argmax(results)
        tag=labels[results_index]
        #print(results[results_index])
        if results[results_index]>0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses= tg["responses"]
            
            print("BOT:",random.choice(responses))
        else:
            print("BOT: I don't quite understand, please ask a different question.")
            
            
chat()