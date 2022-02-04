# Created by otrad at 03/02/2022

from collections import Counter
import numpy as np
from sklearn import metrics
from utils.xml_explorer import read_xml, show, naive_balance
from utils.classifier_with_transformers import do_training, transform_from_problems_to_data_set
from utils.classifier_traditional import TraditionalClassifier


def adapt_data(problems, x_data, y_data):
    """
    Just expanding the data
    :param problems:
    :param x_data:
    :param y_data:
    :return:
    """
    for item in problems:
        x_data.append(item.question + " " + item.answer)
        y_data.append(item.solution)


def main():

    instances, label_to_id = read_xml("data/grade_data.xml")
    np.random.seed(112)
    show(instances, True)
    training_problems, validation_problems, testing_problems = np.split(instances,
                                         [int(.8 * len(instances)),
                                          int(.9 * len(instances))])

    print("Split", len(training_problems), len(validation_problems), len(testing_problems))

    x_train = []
    y_train = []

    adapt_data(training_problems, x_train, y_train)

    x_test = []
    y_test = []
    adapt_data(testing_problems, x_test, y_test)

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
