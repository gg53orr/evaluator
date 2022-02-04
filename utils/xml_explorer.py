# Created by Andres at 02/02/2022
import re
from collections import Counter
import xml.etree.ElementTree as ET
from utils.container_manager import Problem


def get_text(a_key, node):
    """
    get the text of a node's child
    :param a_key:
    :param node:
    :return:
    """
    for child in node:

        if str(child.tag) == a_key:
            return child.text
    return None


def get_solution(child):
    """

    :param child:
    :return:
    """
    for index2, child2 in enumerate(child):
        attri2 = child2.attrib
        if str(child2.tag) == "Annotation":
            annotation_string = child2.attrib.get("Label")
            items = annotation_string.split("|")
            for item in items:
                unit = re.split(r'\(|\)', item)
                if unit[1] == "1":
                    return unit[0]
    return None


def show(instances):

    solution_counter = Counter()

    for instance in instances:
        solution_counter[instance.solution] += 1

    for a_type in solution_counter:
        print(a_type, solution_counter[a_type])


def naive_balance(instances, maxi):

    counter = Counter()
    balanced = []
    for instance in instances:

        if counter[instance.solution] < maxi:
            balanced.append(instance)
        counter[instance.solution] += 1

    return balanced


def read_xml(file_name):
    """
    Just exploring the file
    :param file_name:
    :return:
    """
    tree = ET.parse(file_name)
    root = tree.getroot()
    annotated = 0
    instances = []
    label_to_id_dict = dict()
    id = 0
    for index, child in enumerate(root):

        problem = Problem()
        problem.answer = get_text("Answer", child)
        problem.question = get_text("Question", child)
        problem.description = get_text("ProblemDescription", child)
        problem.set_references(get_text("ReferenceAnswers", child))
        problem.solution = get_solution(child)
        instances.append(problem)
        if not problem.solution in label_to_id_dict:
            id += 1
            label_to_id_dict[problem.solution] = id
    return instances, label_to_id_dict
