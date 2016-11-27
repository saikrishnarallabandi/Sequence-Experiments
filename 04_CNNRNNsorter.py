#!/usr/bin/python

import numpy as np

## Encode the given sequence of numbers into numpy format
def encode(X,seq_len, vocab_size):
    x = np.zeros((len(X),seq_len, vocab_size), dtype=np.float32)
    for ind,batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[ind, j, elem] = 1
    return x

## Data generator for the network
def batch_gen(batch_size=32, seq_len=10, max_no=100):
	x = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)
	y = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)

	while True:
          X = np.random.randint(max_no,size=(batch_size, seq_len))
          Y = np.sort(X, axis=1)

    	  for ind, batch in enumerate(X):
    		for j, elem in enumerate(batch):
    			x[ind,j,elem] = 1

    	  for ind, batch in enumerate(Y):
    	    for j, elem in enumerate(batch):
    	        y[ind,j,elem] = 1

    	  yield x, y
    	  x.fill(0.0)
    	  y.fill(0.0)

def  compute_wer(A,B):
    substitutions = 0
    length = len(A)
    for i in  range(0,len(A)):
        subs = np.count_nonzero(A[i]-B[i])
        substitutions += subs
    return substitutions / (length)         # [Number of sequences and length of  each sequence]  




## Seq to Seq Model

from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dense, TimeDistributedDense, Dropout, TimeDistributedDense
from keras.layers import recurrent, convolutional
from keras.optimizers import RMSprop
#from data import batch_gen, encode
RNN = recurrent.LSTM
Convolution1D = convolutional.Convolution1D

## global param
batch_size = 32
seq_len = 10
max_no = 100

## initialize
model_RNN = Sequential()
model_CNN = Sequential()

print "Initializing Model"

print "Compiling CNN"
model_CNN.add(Convolution1D(64, 3, border_mode='same', input_shape=(seq_len, max_no)))
model_CNN.add(Convolution1D(64, 3, border_mode='same', input_shape=(seq_len, max_no)))
model_CNN.add(TimeDistributedDense(max_no))
model_CNN.add(Dropout(0.25))
model_CNN.add(Activation('softmax'))
model_CNN.compile(loss='categorical_crossentropy', optimizer='adam')
print "Succesfully compiled CNN"

print "Compiling RNN"
model_RNN.add(RNN(100, input_shape=(seq_len,max_no)))
model_RNN.add(Dropout(0.25))
model_RNN.add(RepeatVector(seq_len))
model_RNN.add(RNN(100, return_sequences=True))
model_RNN.add(TimeDistributedDense(max_no))
model_RNN.add(Dropout(0.25))
model_RNN.add(Activation('softmax'))
model_RNN.compile(loss='categorical_crossentropy', optimizer='adam')
print "Succesfully compiled RNN"


for ind, (X,Y) in enumerate(batch_gen(batch_size,seq_len, max_no)):
     
	model_RNN.train_on_batch(X,Y)
        model_CNN.train_on_batch(X,Y)
	if ind % 1000 == 0:
		testX = np.random.randint(max_no,size=(10,seq_len))
		test = encode(testX,seq_len,max_no)
		y_RNN = model_RNN.predict(test, batch_size=10)
                y_CNN = model_CNN.predict(test,batch_size=10)
	        y_RNNsort = np.argmax(y_RNN, axis=2)
	        wer = compute_wer(np.sort(testX), y_RNNsort) 
                print "Error Rate for RNN: " + str(wer)
                y_CNNsort = np.argmax(y_CNN, axis=2)
                wer = compute_wer(np.sort(testX), y_CNNsort) 
                print "Error Rate for CNN: " + str(wer)
                print '\n'

		
#f.close()



