# import dependencies

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import RNN
import numpy as np
import random
import sys
import pdb




filepath = "/Users/Najeeb/ScriptGenerator/text_generator/superhero-films.txt"
data_text = open(filepath).read()


# mapping of characters to integers

chars = sorted(list(set(data_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
vocab = len(chars)
print "Total Characters: ", n_chars
print "Distict characters: ", vocab

# prepare the dataset of input to output pairs encoded as integers
seq_length = 90
data1 = []
data2 = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	data1.append([char_to_int[char] for char in seq_in])
	data2.append(char_to_int[seq_out])


# reshape X to be [samples, time steps, features]
X = numpy.reshape(data1, (n_patterns, seq_length, 1))

# normalization
X = X / float(n_vocab)

# one hot encode the output variable
Y = np_utils.to_categorical(data2)

# define the LSTM model
model = Sequential()
model.add(LSTM(250, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X , Y, epochs=15, batch_size=120)

model.save_weights('/Users/Najeeb/ScriptGenerator/text_generator_700_0.2_700_0.2_100.h5')


# reverse mapping of intergers to chars

int_to_char = dict((i, c) for i, c in enumerate(chars))

# pick random input pattern as a seed sequence

start = np.random.randint(0, len(dataX)-1)
pattern = data1[start]
# generate characters
for i in range(300):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

