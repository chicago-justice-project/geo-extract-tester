"""
Typically run as

python create_validation_data.py > data/validation.txt
"""

import tagnews
import nltk

df = tagnews.load_data()
df = df.loc[df['locations'].apply(bool), :]
for _, row in df.iterrows():
    txt = tagnews.utils.load_data.clean_string(row['bodytext'])
    spans = [loc['cleaned span'] for loc in row['locations'] if 'cleaned span' in loc]
    prev_stop = 0
    for span in spans:
        start, stop = span
        [print('{} {}'.format(word, '0')) for word in nltk.word_tokenize(txt[prev_stop:start])]
        [print('{} {}'.format(word, '1')) for word in nltk.word_tokenize(txt[start:stop])]
        prev_stop = stop
    print('')
