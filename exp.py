## Data Collection
from time import sleep
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import  pandas as pd
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout,GRU

# ## load the dataset
# data=gutenberg.raw('shakespeare-hamlet.txt')
# ## save to a file
# with open('hamlet.txt','w') as file:
#     file.write(data)

##laod the dataset
with open('hamlet.txt','r') as file:
    text=file.read().lower()

## Tokenize the text-creating indexes for words
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1
# print(total_words)

## create inoput sequences
input_sequences=[]
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences ,  max_sequence_len))
# print(input_sequences)


x,y = input_sequences[:,:-1],input_sequences[:,-1]
# print("x:",x)
# print("y:",y)

y=tf.keras.utils.to_categorical(y,num_classes=total_words)
# print("y:",y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


early_stopping = EarlyStopping(monitor = 'val_loss' , patience = 5 , restore_best_weights = True)

model = Sequential()
model.add(Embedding(total_words,100,input_length=max_sequence_len-1))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words , activation = 'softmax'))

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
model.build(input_shape=(None, max_sequence_len-1))
print(model.summary())


history=model.fit(x_train,y_train,epochs=70,validation_data=(x_test,y_test),verbose=1
                #   callbacks=[early_stopping]
                )

## Save the model
model.save("next_word_lstm.h5")