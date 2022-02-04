from collections import Counter
import numpy as np
from utils.xml_explorer import read_xml, show
from utils.classifier_with_transformers import do_training, transform_from_problems_to_data_set


def main():

    instances, label_to_id = read_xml("data/grade_data.xml")
    show(instances)
    np.random.seed(112)

    training_problems, validation_problems, testing_problems = np.split(instances,
                                         [int(.8 * len(instances)),
                                          int(.9 * len(instances))])
    print("Split", len(training_problems), len(validation_problems), len(testing_problems))
    training_data = transform_from_problems_to_data_set(training_problems)
    validation_data = transform_from_problems_to_data_set(validation_problems)
    print("Going to training")
    do_training(training_data, validation_data)


if __name__ == "__main__":
    main()
