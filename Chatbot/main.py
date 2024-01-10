import nltk
# nltk.download('punkt')s
import numpy as np
import random
import tflearn
import tensorflow
import json
from tensorflow.python.framework import ops
import pickle 
# from tensorflow.python.util import 

# from tensorflow.python.util import *


from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

with open("intents.json") as file:
    data=json.load(file)

# print(data["intents"])

try:# this block is coded so that the saved data doesn't run again reducing redundant runs
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)# storing all required 
        #lists in a pickled file for the model
    #data is read first if not found then enters the except block to preprocess data
 
except Exception:#not going enter the block if try block works successfully
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]

    #iterating through different dict key:values i.e., various intents
    for intent in data["intents"]:#intents are tags in the dictionary 
        for pattern in intent["patterns"]:
            word=nltk.word_tokenize(pattern)# this basically seperates each word with a space as a token 
            # also returns a list of tokens 
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    #now we check the vocabulary size of the model without duplicates 

    words=[stemmer.stem(w.lower()) for w in words if w!='?']#convert all words to lower case 
    words=sorted(list(set(words)))# remove duplicates using set() and converting to list which is 
    # sorted 
    labels=sorted(labels)
    # NN only understands numbers 
    # so we use a "bag of words" representing words in any given pattern to train NN
    # Bag of words is called One Hot Encoding 
    # we then create a list of frequencies of occurences of words 

    training=[]
    output=[]

    out_empty=[0 for _ in range(len(labels))]

    #here we are getting our data ready for model creation 

    #doc will get all docs in docs_x
    for x,doc in enumerate(docs_x):
        bag=[]#bag of words i.e., one hot encoded words 

        word=[stemmer.stem(w) for w in doc] # stemming words in docs_x and putting it in a 
        # predefined list 'word'
        for w in words:
            if w in word:# if word exists in current pattern we're looping through
                bag.append(1)# basically we don't care about the frequency of words but just the presence
                # of those particular words 
            else:
                bag.append(0)# if doesn't exist then add 0 to list


        output_row=out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 #checking where that tag is in our list and set
        # value to 1 in output_row 

        # training and output are both one hot encoded
        training.append(bag)
        output.append(output_row)

        with open("data.pickle","wb") as f:
            pickle.dump((words, labels, training, output), f)   

training=np.array(training)
output=np.array(output)

#model creation 

ops.reset_default_graph()# resetting the underline graph data
#input layer 
net = tflearn.input_data(shape=[None,len(training[0])])#length of input data for training 
# to get shape of training data which is dimension of data (rows X columns)
#hidden layer 
net = tflearn.fully_connected(net,8)# 8 neurons for hidden layer 
net = tflearn.fully_connected(net,8)# 8 neurons for hidden layer 
#output layer 
net = tflearn.fully_connected(net,len(output[0]),activation='softmax')# output layer uses softmax activation
#function to find probability distribution 
#in output layer we decide which tag the NN should give the response from to the user
net=tflearn.regression(net,optimizer='adam')#it defines the loss function and
#optimization algorithm used for training the network.
#basically shows the error measurement between target and actual output 

#now we train our model 
model=tflearn.DNN(net)

try:#if model exists then just load model else create the model 
    model.load("model.tflearn")

except Exception:
    #with more intents and tags we can add more hidden layers but 2 is enough for the given problem
    #now we fit the model i.e., passing in training data to model 
    model.fit(training, output, n_epoch=1000,batch_size=8,show_metric=True)
    #n_epoch means how many times the model sees the data to get trained 
    model.save("model.tflearn")
    #saving the model as "model.tflearn"

# this model takes in a sentence input from user and puts it in a bag of words  
def bag_of_words(s, words):
    bag=[0 for _ in range(len(words))]#creates a blank bag of words list 
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]#stemming words in lower case 

    for se in s_words:
        for i, w in enumerate(words):
            if w==se:#checking if word in words list is equal to current word in sentence 
                bag[i]=1 #if present append 1 to list for that respective word 

    return np.array(bag)

# NOTE : Each neuron represents each specific class
def chat():
    print("You can start chatting with bot")
    print("To exit type exit")
    while(True):
        inp=input("You: ")
        if inp.lower()=="exit":
            print("BOT: See you later!")
            break
        
        # [0] means accessing the first value which here is 1st list in the list
        # containing predicted values
        results= model.predict([bag_of_words(inp,words)])[0]#passing user input as a list of bag of words 
        # print(results)# prints out the probability of each neuron representing a class 
        #here class means which tag it is 
        #so what we do is find the highest probability neuron 
        #argmax() - picking index of largest number
        results_index=np.argmax(results)# argmax gets the max value index neuron 
        # using this index we can know which response should be displayed 
        tag=labels[results_index]#getting the value of the index  which is the label 
        # print(tag)
        if results[results_index] > 0.7:#checking if model has confidence i.e., >70% 
            for tg in data['intents']:
                if tg['tag']==tag:
                    responses=tg['responses']
            print("BOT: ",random.choice(responses))
        
        else:
            print("BOT: I didn't quite understand. Please try again.")        

        
chat()













