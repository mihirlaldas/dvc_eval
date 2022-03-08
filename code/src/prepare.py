import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import os
import random
import re
import sys
from pickle import dump

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data_file = sys.argv[1]
df_original = pd.read_csv(data_file)
print(df_original.head())

columns = ['Text', 'Cat1']

df = shuffle(df_original[columns])
p = re.compile(r'[^\w\s]+')

df['Text'] = [p.sub('', x) for x in df['Text'].tolist()]
df.apply(lambda x: x.astype(str).str.lower())

x,y = df.Text, df.Cat1
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

# train_x.to_csv('data/prepared/train_x.csv', encoding='utf-8')
# test_x.to_csv('data/prepared/test_x.csv', encoding='utf-8')
# train_y.to_csv('data/prepared/train_y.csv', encoding='utf-8')
# test_y.to_csv('data/prepared/test_y.csv', encoding='utf-8')

dump(train_x, open('data/prepared/train_x.pkl', 'wb'))
dump(test_x, open('data/prepared/test_x.pkl', 'wb'))

dump(train_y, open('data/prepared/train_y.pkl', 'wb'))
dump(test_y, open('data/prepared/test_y.pkl', 'wb'))
dump(x, open('data/prepared/original_x.pkl', 'wb'))
dump(y, open('data/prepared/original_y.pkl', 'wb'))