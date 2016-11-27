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
    return substitutions / length   




## Seq to Seq Model

from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dense, TimeDistributedDense, Dropout, TimeDistributedDense
from keras.layers import recurrent, convolutional
from keras.optimizers import RMSprop
#from data import batch_gen, encode
#RNN = recurrent.LSTM
Convolution1D = convolutional.Convolution1D

## global param
batch_size = 32
seq_len = 10
max_no = 100

## initialize
model = Sequential()

print "Initializing Model"

## Encoder
#model.add(TimeDistributedDense(100, input_shape=(seq_len,max_no)))
model.add(Convolution1D(64, 3, border_mode='same', input_shape=(seq_len, max_no)))

# Activation
#model.add(Activation('relu'))

# Dropout
#model.add(Dropout(0.25))

print "Added first layer"

#model.add(TimeDistributedDense(3*max_no))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))
model.add(Convolution1D(64, 3, border_mode='same', input_shape=(seq_len, max_no)))
print "Added second layer"


#model.add(TimeDistributedDense(5*max_no))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))
#print "Added third layer"

#model.add(TimeDistributedDense(3*max_no))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))
#print "Added fourth layer"

# Linear Layer
model.add(TimeDistributedDense(max_no))

# Non linear on top of  each linear layer
model.add(Dropout(0.25))
model.add(Activation('softmax'))

print "Added softmax layer"

rms =  RMSprop()
print "Compiling Model"
model.compile(loss='categorical_crossentropy', optimizer='adam')

#f = open('RNNsorting.log','w')
# Training Loop
for ind, (X,Y) in enumerate(batch_gen(batch_size,seq_len, max_no)):
        #print X
        #print Y
	model.train_on_batch(X,Y)
	#print ind
	if ind % 1000 == 0:
		testX = np.random.randint(max_no,size=(10,seq_len))
		test = encode(testX,seq_len,max_no)
		#print testX
		y = model.predict(test, batch_size=10)
		#print "Actual sorted  output "  
		#print np.sort(testX)
		#print "DNN sorting "
                y_sort = np.argmax(y, axis=2)
		#print y_sort
                wer = compute_wer(np.sort(testX), y_sort) 
                print "Error Rate: " + str(wer)
		
#f.close()



