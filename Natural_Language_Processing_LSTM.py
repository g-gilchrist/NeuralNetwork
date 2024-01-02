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
import sys


#-----------------------------------------------
# Loads txt file
#-----------------------------------------------
# load ascii text and covert to lowercase

def load(filename):
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    return raw_text


#-----------------------------------------------
# Preprocess the data
#-----------------------------------------------
#This breaks the text apart into characters and assigns numerical values to each charachter

def preprocess(raw_text, seq_length):
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text))) # creates a sorted list of all chararchters in the raw text and removes any duplicates
    char_to_int = dict((c, i) for i, c in enumerate(chars)) # assigns a numerical value to each charachter
    int_to_char = dict((i, c) for i, c in enumerate(chars)) # assigns a charachter to each numerical value
    
    # summarize the loaded data
    n_chars = len(raw_text) # length of text in charachters 
    n_vocab = len(chars) # length of unique charachters utilized in the text 
    print(f"Total Characters: {n_chars}") # outputs to console the amount of charachters in the document
    print(f"Total Unique Characters: {n_vocab}") # outputs to console the amount of unique charachters used in the document

    # prepare the dataset of input to output pairs encoded as integers
    dataX = [] # declares a list to store sequence in digits (first 100 chars)
    dataY = [] # declares a list to store sequence out digits (> first 100 chars)
    # this splits up the text into sequences of 100 charachters
    for i in range(0, n_chars - seq_length, 1): # creates a beiginning point of 0 and an end point of the number of charachters minus the sequence length, it increments the for loop by one.
        seq_in = raw_text[i:i + seq_length] # slices the text into 100 charachters
        seq_out = raw_text[i + seq_length] # slices the text into individual charachters after the first 100 charachters
        dataX.append([char_to_int[char] for char in seq_in]) # converts the char from seq_in to an int and appends it to dataX
        dataY.append(char_to_int[seq_out]) # converts the char from seq_out to an int and appends it to dataY
    n_patterns = len(dataX) # stores the length of dataX, which will be 100 less than total charachters as the first 100 charachters has nothing to compare it to
    print("Total Patterns: ", n_patterns) # outputs the number of patterns to the console
    
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    
    # normalize to avoid bias by deviding X by the number of unique chars
    X = X / float(n_vocab)
    
    # one hot encode the output variable, higher numbers are not more important, they are a numerical representation of a char
    y = utils.to_categorical(dataY) #(> first 100 chars)


    # returns the variables to be used in other functions
    return X,y,char_to_int,int_to_char,dataX,n_vocab


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

def lstm(X,y,dataX,int_to_char):
    
    # define the LSTM model
    model = Sequential() # creates a sequential object from the keras.model library
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)) # adds a long short-term memory object to the model, this creates a layer for the neural network
    model.add(Dropout(0.2)) # creates a dropout layer for the neural network, this reduces overfitting by ignoring neurons
    model.add(LSTM(256))  # adds another long short-term memory object to the model, this creates a layer for the neural network
    model.add(Dropout(0.2)) # creates another dropout layer for the neural network, this reduces overfitting by ignoring neurons
    model.add(Dense(y.shape[1], activation='softmax')) # Layer of neurons receiving data from all neurons of previous layer
    
    # load the network weights
    filename = "weights-improvement-50-1.1959-bigger.keras" # stores the 35th epoch into the filename variable
    filepath = f"./checkpoints/{filename}" # the checkpoints are in the checkpoint folder
    model.load_weights(filepath) # loads the 35th epoch of our LTSM hdf5 file into our model
    model.compile(loss='categorical_crossentropy', optimizer='adam') # defines the loss function, optimizers, and metrics necessary for predictions
    
    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1) # creates a random number
    pattern = dataX[start] # the random number becomes the index, dataX is a list of numerals representing a unique charachter used in the document
    print("\n\n------------------------------------------------------------------------")
    print("\n\nSeed: \n")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"", '\n') # converts the integers from the pattern back into charachters
    
    # returns the variables to be used in other functions
    return pattern,model
    

#-----------------------------------------------
# Text Generator
#-----------------------------------------------
# This generates text based on the neural network predictions
    
def txtGen(pattern, n_vocab, model, int_to_char):  
    print("\n\n------------------------------------------------------------------------")
    print("\n\nGenerated Text: \n")
    # generate characters
    for char in range(1000): # generates 1000 charachter prediction
        x = numpy.reshape(pattern, (1, len(pattern), 1)) # returns an array with the same data in a different dimension, in this case a 3d array [[pattern,...,len(patern)]]
        x = x / float(n_vocab) # deviding the patern by the number of unique charachters 
        prediction = model.predict(x, verbose=0) # making a prediction based on our pattern
        index = numpy.argmax(prediction) # finds the most probable prediction
        result = int_to_char[index] # converts the integers from the most probable prediction back to a charachter
        seq_in = [int_to_char[value] for value in pattern]  # converts the integers from the pattern back to a charachter
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
    X,y,char_to_int,int_to_char,dataX,n_vocab = preprocess(raw_text,100)
    
    # parameters: input, target, epoch, batch_size
    #checkpoint(X,y,50,64) # Only use this when you want to train the neural network
    
    print("Original: \n")
    print(raw_text)
    
    pattern,model = lstm(X,y,dataX,int_to_char)
    
    txtGen(pattern,n_vocab,model,int_to_char)
    
main()

