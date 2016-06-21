import multiprocessing
import os
import re
import sys

import nltk
import slate


def construct_path_list(starting_path, extension):
    path_list = []
    for root, dirs, files in os.walk(starting_path):
        for filename in files:
            if filename.lower().endswith(extension):
                path = os.path.join(root, filename)
                path_list.append(path)
    return path_list


def sanitise(text):
    remove_non_words = re.sub(r"[^A-Za-z \n]", "", text)
    remove_surplus_spaces = re.sub(r"[\n ]+", " ", remove_non_words)

    return [i for i in remove_surplus_spaces.split(" ") if len(i) > 1]


def filter_stopwords(document):
    stopwords = nltk.corpus.stopwords.words("english")
    return [i for i in document if i not in stopwords]


def lemmatize(document):
    lmt = nltk.stem.WordNetLemmatizer()
    return [lmt.lemmatize(i) for i in document]


def process_document(path):
    print("Processing {file}".format(file=path))
    field, subject, _, fname = path.split("/")[-4:]

    with open(path, "r") as in_file:
        text = sanitise(in_file.read())

    doc = lemmatize(filter_stopwords(text))
    out_path = os.path.join(OUTPUT_DIR, field, subject)

    print(out_path)

    try:
        os.makedirs(out_path)
    except OSError:
        pass

    with open(os.path.join(out_path, fname), "w") as out_file:
        out_file.write("\n".join(doc))


if __name__ == "__main__":
    OUTPUT_DIR = "../output/"

    paths = construct_path_list(sys.argv[1], ".txt")

    pool = multiprocessing.Pool(3)
    pool.map_async(func=process_document, iterable=paths)
    pool.close()
    pool.join()
