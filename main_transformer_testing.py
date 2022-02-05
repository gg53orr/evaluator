"""
Script for testing transformer model for classifying
"""
from collections import Counter
import numpy as np
from core.xml_explorer import read_xml, show
from core.classifier_with_transformers import do_testing, transform_from_problems_to_data_set


def main():

    instances, label_to_id = read_xml("data/grade_data.xml")
    show(instances)
    np.random.seed(112)

    _, _, testing_problems = np.split(instances,
                                         [int(.8 * len(instances)),
                                          int(.9 * len(instances))])
    testing_data = transform_from_problems_to_data_set(testing_problems)

    print("Do testing")
    do_testing(testing_data)


if __name__ == "__main__":
    main()
