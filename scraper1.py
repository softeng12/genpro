import nltk
import tensorflow.keras as keras
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import random
import json
import tensorflow
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD
nltk.download('wordnet')



lemmatizer=WordNetLemmatizer()
lr = 0.01
# intents=json.load(open('G:\data.json').read())
with open('G:\data.json') as file:
    intents=json.load(file)
words=[]
classes=[]
documents=[]
ignore_letter=['?','!','-',',']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letter]
words=sorted(set(words))
classes=sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)
for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# sgd=SGD(lr=lr,decay=1e-6,momentum=0.9,nesterov=True), optimizer=sgd


# model.compile(loss='categorical_crossentropy')
model.compile(loss='categorical_crossentropy' ,metrics=['accuracy'])
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbotmodel.h5',hist)
print("Done")





# stemmer=LancasterStemmer()

# with open("G:\data.json") as file:
#     data=json.load(file)

# try:
#     with open("data.pickle",'rb') as f:
#         words,labels,training,output=pickle.load(f)

# except:
#     words=[]
#     labels=[]
#     docs_x=[]
#     docs_y=[]

#     for intent in data["intents"]:
#         for pattern in intent["patterns"]:
#             wrds=nltk.word_tokenize(pattern)
#             words.extend(wrds)
#             docs_x.append(wrds)  
#             docs_y.append(intent["tag"])
            
            
#         if intent["tag"] not in labels:
#             labels.append(intent["tag"])
            
#     words=[stemmer.stem(w.lower()) for w in words if w != "?"]
#     words=sorted(list(set(words)))
#     labels=sorted(labels)
#     training=[]
#     output=[]
#     out_empty=[0 for _ in range(len(labels))]
#     for x ,doc in enumerate(docs_x):
#         bag=[]
#         wrds=[stemmer.stem(w) for w in doc ]
#         for w in words:
#             if w in wrds:
#                 bag.append(1)
#             else:
#                 bag.append(0)
                
#         output_row=out_empty[:]
#         output_row[labels.index(docs_y[x])]=1
#         training.append(bag)
#         output.append(output_row)
        
#     training=np.array(training)
#     output=np.array(output)
    
#     with open("data.pickle",'wb') as f:
#         pickle.dump((words,labels,training,output),f)
# # tensorflow.reset_default_graph()
# net=tflearn.input_data(shape=[None,len(training[0])])
# net=tflearn.fully_connected(net,5)
# net=tflearn.fully_connected(net,5)
# net=tflearn.fully_connected(net,len(output[5]),activation="softmax")
# model=tflearn.DNN(net)
# try:
#     model.load("model.tflearn")
# except:
#     model.fit(training,output,n_epoch=100,show_metric=True)
#     model.save("model.tflearn")
    
    
# def bag_of_words(s,words):
#     bag=[0 for _ in range(len(words))]
#     s_words=nltk.word_tokenize(s)
#     s_words=[stemmer.stem((word.lower())) for word in s_words]
#     for se in s_words:
#         for i,w in enumerate(words):
#             if w==se:
#                 bag[i]=(1)
#     return numpy.array(bag)

# def chat():
#     print("start talking with the boat(type quit to stop)!")
#     while True:
#         inp=input("you: ")
#         if inp.lower()=="quit":
#             break
        
#         results=model.predict([bag_of_words(inp,words)])
#         results_index=numpy.argmax(results)
#         tag=labels[results_index]
#         for tg in data["intents"]:
#             if tg["tag"]==tag:
#                 responses=tg['responses']
#                 print(random.choice(responses))
        
        
# chat()
# # print(data["intents"])