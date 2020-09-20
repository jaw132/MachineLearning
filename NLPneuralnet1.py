#!wget - -no - check - certificate \
 #   https: // storage.googleapis.com / laurencemoroney - blog.appspot.com / bbc - text.csv \
  #            - O / tmp / bbc - text.csv

import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
# Convert it to a Python list and paste it here

sentences = []
labels = []
with open("/tmp/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        row_list = row[1].split()
        clean_list = []
        for word in row_list:
            if word not in stopwords:
                clean_list.append(word)

        labels.append(row[0])
        sentences.append(" ".join(x for x in clean_list))

print(len(sentences))
print(sentences[0])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post")
print(padded[0])
print(padded.shape)

label_token = Tokenizer()
label_token.fit_on_texts(labels)
label_word_index = label_token.word_index
label_seq = label_token.texts_to_sequences(labels)
print(label_seq)
print(label_word_index)

