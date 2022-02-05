# Created by Andres at 05/02/2022

from collections import Counter
import numpy as np
from sklearn import metrics
import spacy

from core.xml_explorer import read_xml, show, naive_balance
from core.classifier_with_transformers import do_training, transform_from_problems_to_data_set
from core.classifier_traditional import TraditionalClassifier
from core.semantic_support import build_w2v
nlp = spacy.load("en_core_web_sm")


def is_semantic(token):
    """
    Basic filter. Consider using w2v
    :param token:
    :return:
    """
    if token.pos_ == "NOUN" or token.pos_ == "VERB" or \
            token.pos_ == "ADV" or token.pos_ == "ADP" or token.pos_ == "NUM":
        return True
    return False


def simplify(text):

    simplified_text = []
    doc = nlp(text)
    for token in doc:
        simplified_text.append(str(token.lemma))
    return " ".join(simplified_text)


def adapt_data(problems, x_data, y_data, corpus):
    """
    Just expanding the data
    :param problems:
    :param x_data:
    :param y_data:
    :param corpus:
    :return:
    """
    for item in problems:

        x_data.append(simplify(item.question) + " > " + simplify(item.answer))
        y_data.append(item.solution)
        if corpus is not None:
            corpus.append(item.question)
            corpus.append(item.answer)
            for reference in item.references:
                corpus.append(reference)


def main():

    instances, label_to_id = read_xml("data/grade_data.xml")
    np.random.seed(112)
    show(instances)
    training_problems, validation_problems, testing_problems = np.split(instances,
                                         [int(.8 * len(instances)),
                                          int(.9 * len(instances))])

    print("Split", len(training_problems), len(validation_problems), len(testing_problems))

    x_train = []
    y_train = []
    corpus = []
    adapt_data(training_problems, x_train, y_train, corpus)
    adapt_data(validation_problems, x_train, y_train, corpus)

    # We could use a word2vec from corpus
    print(len(corpus), " amount of sentences for w2v")
    build_w2v(corpus, "data/")
    # The w2v here can be used to produce more examples
    x_test = []
    y_test = []
    adapt_data(testing_problems, x_test, y_test, None)

    classifier = TraditionalClassifier()
    path = "data/teacher.joblib"
    print("Total training:", len(x_train))
    classifier.fit(x_train, y_train, path)

    predicted = classifier.predict(x_test)
    prediction = np.mean(predicted == y_test)
    print(prediction, " prediction")
    the_labels = list(label_to_id.keys())

    print(metrics.classification_report(y_test,
                                        predicted,
                                        target_names = the_labels))



if __name__ == "__main__":
    main()
