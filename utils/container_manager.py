# Created by otrad at 02/02/2022
from typing import List
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class Problem:
    """
    The whole record
    """

    def __init__(self):
        self.question = None
        self.answer = None
        self.solution = None
        self.description = None
        self.references = []

    def get_input(self):
        """

        :return:
        """
        return self.question + " > " + self.answer

    def set_references(self, reference):
        """
        Adapt the references as list of strings
        :param reference:
        :return:
        """
        lines = reference.split("\n")
        for line in lines:
            if len(line.strip()) > 1:
                start = line.find(":") + 2
                self.references.append(line[start:].strip())


class DataSet(torch.utils.data.Dataset):

    def __init__(self, texts, labels):
        # 'business' : 1 etc
        self.texts = texts
        self.labels = labels

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        """
        Get a batch of labels
        :param idx:
        :return:
        """
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        """
        Get a batch of texts
        :param idx:
        :return:
        """
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
