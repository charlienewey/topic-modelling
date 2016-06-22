import logging
import sys

import gensim

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO
)

DICT = sys.argv[1]
MM_FILE = sys.argv[2]

id2word = gensim.corpora.Dictionary.load(DICT)
mm = gensim.corpora.MmCorpus(MM_FILE)

# get the LDA model
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=200, passes=1)
lda.print_topics(50)
