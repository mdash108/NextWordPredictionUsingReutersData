# NextWordPredictionUsingReutersData
In this repo I show one way of developing a next word prediction python code using Reuters data

```python
# Import the Libraries
import nltk
from nltk.corpus import reuters
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
import re
from sklearn.model_selection import train_test_split
```

```python
# set sentence limit: otherwise leads to memory error in devices with less memory
sentCntLimit = 2000
```
```python
# create word mapping
sentCnt = 0
uniq_words = set()
for sentence in reuters.sents():
    if sentCnt<sentCntLimit:
        sentCnt += 1
    else:
        break
    # clean the text: lower case, only alphabet
    sentence = [word.lower() for word in sentence]
    sentence = [re.sub("[^a-zA-Z]", '', word) for word in sentence]
    sentence = [re.sub(r"'s/b", '', word) for word in sentence]
    # remove empty string
    for word in sentence:
        if word == '':
            sentence.remove(word)
    # replace 'u' followed by 's' by 'us'
    i=0
    while (1):
        if (i<len(sentence)-1):
          if (sentence[i]=='u') & (sentence[i+1]=='s'):
            sentence.pop(i)
            sentence.pop(i)
            sentence.insert(i, 'usa')
        if (i>=len(sentence)-1):
            break
        else:
            i+=1
    # now go thru the words in sentence and add the new ones to uniq_words set
    for word in sentence:
        uniq_words.add(word)
        #print(uniq_words)
        #x1=input("Enter something: ")
uniq_words = sorted(uniq_words)
mapping = dict((word, ind) for ind, word in enumerate(uniq_words))
```
```python
# create sequences
sentCnt = 0
all_seq = []
for sentence in reuters.sents():
    if sentCnt<sentCntLimit:
        sentCnt += 1
    else:
        break
    sentence = [word.lower() for word in sentence]
    sentence = [re.sub("[^a-zA-Z]", '', word) for word in sentence]
    sentence = [re.sub(r"'s/b", '', word) for word in sentence]
    # remove empty string
    for word in sentence:
        if word == '':
            sentence.remove(word)
    # replace 'u' followed by 's' by 'us'
    i=0
    while (1):
        if (i<len(sentence)-1):
          if (sentence[i]=='u') & (sentence[i+1]=='s'):
            sentence.pop(i)
            sentence.pop(i)
            sentence.insert(i, 'usa')
        if (i>=len(sentence)-1):
            break
        else:
            i+=1

    # create sequences of 2 words to all words and append them to all_seq
    for i in range(2, len(sentence)):
        seq1 = []
        for w1 in sentence[:i]:
            seq1.append(mapping[w1])
        all_seq.append(seq1)

# find the max_len among all sequences in all_seq
max_len = max(len(all_seq[i]) for i in range(len(all_seq)))

# pre-pad all sequences for length max_len
all_seq = pad_sequences(all_seq, maxlen=max_len, truncating='pre')

# create X and y from all_seq
all_seq = np.array(all_seq)
X = all_seq[:, :-1]
y = all_seq[:, -1]

# use to_categorical to convert each y value to a vector of length of uniq_words
y = to_categorical(y, len(uniq_words))

# split X, y into training and validation
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state = 42)
```
```python
# create a LSTM model: 1st layer is Embedding, 2nd is LSTM, 3rd is a Dense layer with softmax
model = Sequential()
model.add(Embedding(len(uniq_words), 50, input_length = max_len-1, trainable=True))
model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(len(uniq_words), 'softmax'))
model.compile(loss="categorical_crossentropy", metrics='acc', optimizer='adam')
model.fit(X_tr, y_tr, epochs=40, verbose=1, validation_data=(X_val, y_val))
```
```python
# predict text
seed_text = "asian exporters"
seed_words = seed_text.split()
cntPred = 5
for i in range(cntPred):
    # map seed_text using mapping
    map_words = [mapping[w1] for w1 in seed_words]
    # use pad_sequences to pad 0s to map_words
    pad_input = pad_sequences([map_words], maxlen=max_len-1, truncating='pre')
    # predict
    yhat_probs = model.predict(pad_input, verbose=1)
    # get the index of the max prob among yhat_probs
    yhat = np.argmax(yhat_probs)
    for word, ind1 in mapping.items():
        if yhat==ind1:
            seed_words.append(word)
            break
print(seed_words)
```
