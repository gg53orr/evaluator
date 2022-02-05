# Created by Andres at 05/02/2022
import os
import spacy
from gensim.models.word2vec import Word2Vec, LineSentence
from pprint import pprint
from pathlib import Path
from copy import deepcopy
from multiprocessing import cpu_count


def build_w2v(texts, output_dir):
    """
    Build a simple word2vec
    :param texts:
    :param output_dir:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    st_sentences = []
    w2v_file = os.path.join(output_dir, 'semantics.w2v')

    for text in texts:
        doc = nlp(text)
        for sentence in doc.sents:
            st_sentence = []
            for token in sentence:
                token_text = token.text
                st_sentence.append(token_text)

            st_sentences.append(st_sentence)

    model = Word2Vec(st_sentences, min_count=0, workers=cpu_count())
    new_model = deepcopy(model)
    new_model.save(w2v_file)


