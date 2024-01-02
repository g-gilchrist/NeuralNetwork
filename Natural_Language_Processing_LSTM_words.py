# Grant Gilchrist
# 06/19/2022 
# CS379-2203A-01 - Machine Learning 
# This is a Recurrent Neural Network utilizing Long Short-Term Memory on Lewis Carroll's
# literary classic Alice in Wonderland.
# Algorithm from Fundamentals of Supervised Machine Learning: With Applications in Python, R, and Stata by: Giovanni Cerulli
# For other free ebooks to train on, try gutenberg.org

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras import utils    
import sys, re


#-----------------------------------------------
# Loads txt file
#-----------------------------------------------
# load ascii text and covert to lowercase

def load(filename):
    raw_text = open(filename, 'r', encoding='utf-8').read()
    #raw_text = raw_text.lower()
    return raw_text


#-----------------------------------------------
# split text into sentences
#-----------------------------------------------
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    # -*- coding: utf-8 -*-
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "'" in text: text = text.replace("`","'")
    if "”" in text: text = text.replace(".”","”.")
    if "'" in text: text = text.replace(".'","'.")
    if "'" in text: text = text.replace(".)",").")
    if "”" in text: text = text.replace("!”","”!")
    if "'" in text: text = text.replace("!'","'!")
    if "'" in text: text = text.replace("!)",")!")
    if "”" in text: text = text.replace("?”","”?")
    if "'" in text: text = text.replace("?'","'?")
    if "'" in text: text = text.replace("?)",")?")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    for index, sentence in enumerate(sentences):
        if "”" in sentence: sentences[index] = sentences[index].replace("”.",".”")
        if "'" in sentence: sentences[index] = sentences[index].replace("'.",".'")
        if ")" in sentence: sentences[index] = sentences[index].replace(").",".)")
        if "”" in sentence: sentences[index] = sentences[index].replace("”!","!”")
        if "'" in sentence: sentences[index] = sentences[index].replace("'!","!'")
        if ")" in sentence: sentences[index] = sentences[index].replace(")!","!)")
        if "”" in sentence: sentences[index] = sentences[index].replace("”?","?”")
        if "'" in sentence: sentences[index] = sentences[index].replace("'?","?'")
        if ")" in sentence: sentences[index] = sentences[index].replace(")?","?)")
        if sentences[index][0].isupper() or sentences[index][1].isupper() == False: 
            sentences[index - 1] = sentences[index - 1] + sentence
            sentences.pop(index)
    return sentences

#-----------------------------------------------
# Preprocess the data
#-----------------------------------------------
#This breaks the text apart into words and assigns numerical values to each word

def preprocess(raw_text, seq_length):
    sentence_list = split_into_sentences(raw_text)
    word_list = [re.split('( )', sentence) for sentence in sentence_list]
    # create mapping of unique words to integers, and a reverse mapping
    words = sorted(list(set(word_list))) # creates a sorted list of all  words in the raw text and removes any duplicates
    word_to_int = dict((c, i) for i, c in enumerate(words)) # assigns a numerical value to each word
    int_to_word = dict((i, c) for i, c in enumerate(words)) # assigns a word to each numerical value
    
    # summarize the loaded data
    n_words = len(word_list) # length of text in words 
    n_vocab = len(words) # length of unique words utilized in the text 
    print(f"Total Words: {n_words}") # outputs to console the amount of words in the document
    print(f"Total Unique Words: {n_vocab}") # outputs to console the amount of unique words used in the document

    # prepare the dataset of input to output pairs encoded as integers
    dataX = [] # declares a list to store sequence in digits (first 100 words)
    dataY = [] # declares a list to store sequence out digits (> first 100 words)
    # this splits up the text into sequences of 100 words
    for i in range(0, n_words - seq_length, 1): # creates a beiginning point of 0 and an end point of the number of words minus the sequence length, it increments the for loop by one.
        seq_in = word_list[i:i + seq_length] # slices the text into 100 words
        seq_out = word_list[i + seq_length] # slices the text into individual words after the first 100 words
        dataX.append([word_to_int[word] for word in seq_in]) # converts the word from seq_in to an int and appends it to dataX
        dataY.append(word_to_int[seq_out]) # converts the word from seq_out to an int and appends it to dataY
    n_patterns = len(dataX) # stores the length of dataX, which will be 100 less than total words as the first 100 words has nothing to compare it to
    print("Total Patterns: ", n_patterns) # outputs the number of patterns to the console
    
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    
    # normalize to avoid bias by deviding X by the number of unique words
    X = X / float(n_vocab)
    
    # one hot encode the output variable, higher numbers are not more important, they are a numerical representation of a word
    y = utils.to_categorical(dataY) #(> first 100 words)


    # returns the variables to be used in other functions
    return X,y,word_to_int,int_to_word,dataX,n_vocab


#-----------------------------------------------
# create check points 
#-----------------------------------------------
# This creates a checkpoint for each epoch that is used to train the neural network

def checkpoint(input, target, number_of_iterations, number_of_samples): # passing in the variables created in preprocessing to train the neural network 
    
    # define the LSTM model
    model = Sequential() # creates a sequential object from the keras.model library
    model.add(LSTM(256, input_shape=(input.shape[1], input.shape[2]), return_sequences=True)) # adds a long short-term memory object to the model, this creates a layer for the neural network
    model.add(Dropout(0.2)) # creates a dropout layer for the neural network, this reduces overfitting by ignoring neurons
    model.add(LSTM(256)) # adds another long short-term memory object to the model, this creates a layer for the neural network
    model.add(Dropout(0.2)) # creates a dropout layer for the neural network, this reduces overfitting by ignoring neurons
    model.add(Dense(target.shape[1], activation='softmax')) # Layer of neurons receiving data from all neurons of previous layer
    model.compile(loss='categorical_crossentropy', optimizer='adam') # defines the loss function, optimizers, and metrics necessary for predictions
    
    # define the checkpoint
    filepath="./checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"  # Creates a unique filename
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min') # saves the model after each epoch with the unique filename if the loss has improved
    callbacks_list = [checkpoint] # creates a list of checkpoints
    
    # fit the model
    # epochs = number of passes, batch size = amount of samples
    model.fit(
        x=input, 
        y=target, 
        epochs=number_of_iterations, 
        batch_size=number_of_samples, 
        callbacks=callbacks_list
    )


#-----------------------------------------------
# Long Short Term Memory Algorithm
#-----------------------------------------------
# This utilizes the checkpoint created to train the nerual network for patterns

def lstm(X,y,dataX,int_to_word):
    
    # define the LSTM model
    model = Sequential() # creates a sequential object from the keras.model library
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)) # adds a long short-term memory object to the model, this creates a layer for the neural network
    model.add(Dropout(0.2)) # creates a dropout layer for the neural network, this reduces overfitting by ignoring neurons
    model.add(LSTM(256))  # adds another long short-term memory object to the model, this creates a layer for the neural network
    model.add(Dropout(0.2)) # creates another dropout layer for the neural network, this reduces overfitting by ignoring neurons
    model.add(Dense(y.shape[1], activation='softmax')) # Layer of neurons receiving data from all neurons of previous layer
    
    # load the network weights
    filename = "weights-improvement-50-3.0802-bigger.keras" # stores the 35th epoch into the filename variable
    filepath = f"./checkpoints/Alice_in_Wonderland/words/128-batch-size/{filename}" # the checkpoints are in the checkpoint folder
    model.load_weights(filepath) # loads the 35th epoch of our LTSM hdf5 file into our model
    model.compile(loss='categorical_crossentropy', optimizer='adam') # defines the loss function, optimizers, and metrics necessary for predictions
    
    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1) # creates a random number
    pattern = dataX[start] # the random number becomes the index, dataX is a list of numerals representing a unique word used in the document
    print("\n\n------------------------------------------------------------------------")
    print("\n\nSeed: \n")
    print("\"", ''.join([int_to_word[value] for value in pattern]), "\"", '\n') # converts the integers from the pattern back into words
    
    # returns the variables to be used in other functions
    return pattern,model
    

#-----------------------------------------------
# Text Generator
#-----------------------------------------------
# This generates text based on the neural network predictions
    
def txtGen(pattern, n_vocab, model, int_to_word):  
    print("\n\n------------------------------------------------------------------------")
    print("\n\nGenerated Text: \n")
    # generate words
    for word in range(1000): # generates 1000 word prediction
        x = numpy.reshape(pattern, (1, len(pattern), 1)) # returns an array with the same data in a different dimension, in this case a 3d array [[pattern,...,len(patern)]]
        x = x / float(n_vocab) # deviding the patern by the number of unique words 
        prediction = model.predict(x, verbose=0) # making a prediction based on our pattern
        index = numpy.argmax(prediction) # finds the most probable prediction
        result = int_to_word[index] # converts the integers from the most probable prediction back to a word
        seq_in = [int_to_word[value] for value in pattern]  # converts the integers from the pattern back to a word
        sys.stdout.write(result) # outputs the results to the console
        pattern.append(index) # appends the index to the pattern array
        pattern = pattern[1:len(pattern)] # slices the pattern array and overwrites itself
    print("\nDone.")


#-----------------------------------------------
# Main
#-----------------------------------------------
# simple main function that calls the different functions.

def main():
    
    raw_text = load("alice_in_wonderland.txt")
    X,y,word_to_int,int_to_word,dataX,n_vocab = preprocess(raw_text,100)
    
    # parameters: input, target, epoch, batch_size
    # checkpoint(X,y,50,128) # Only use this when you want to train the neural network
    
    print("Original: \n")
    print(raw_text)
    
    pattern,model = lstm(X,y,dataX,int_to_word)
    
    txtGen(pattern,n_vocab,model,int_to_word)
    
main()

