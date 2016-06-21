import collections
import os
import sys

import gensim


IN_PATH = sys.argv[1]

corpora = []
for root, dirs, files in os.walk(IN_PATH):
    for f in files:
        if f.endswith(".txt"):
            path = os.path.join(root, f)
            with open(path, "r") as _in:
                document = [item for item in _in.read().split("\n") if len(item) > 0]

            if len(document) == 0:
                continue

            corpora.append(document)

frequency = collections.Counter([item for sublist in corpora for item in sublist])

corpus_dictionary = gensim.corpora.Dictionary(corpora)
corpus_dictionary.save(os.path.join(IN_PATH, "corpus.dict"))

tokenized = [corpus_dictionary.doc2bow(text) for text in corpora]
gensim.corpora.MmCorpus.serialize(os.path.join(IN_PATH, "corpus.mm"), tokenized)
