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
from sklearn.model_selection import train_test_split```

```python
# set sentence limit: otherwise leads to memory error in devices with less memory
sentCntLimit = 2000```
