# Created by otrad at 03/02/2022
from typing import List
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from transformers import BertTokenizer
from transformers import BertModel

from utils.container_manager import Problem, DataSet
from utils.bert_trainer import BertTrainer


def transform_from_problems_to_data_set(items : List[Problem]) -> DataSet:

    labels = []
    texts = []
    labels_dict = {'correct': 0,
              'correct_but_incomplete': 1,
              'contradictory': 2,
              'incorrect': 3
              }

    for item in items:
        id_label = labels_dict[item.solution]
        labels.append(item.solution)
        texts.append(item.get_input())
    return DataSet(texts, labels)


def do_training(training: DataSet, validation: DataSet):

    the_trainer = BertTrainer()
    the_trainer.create_model()
    the_trainer.train(training.texts, training.labels,
                      validation.texts, validation.labels)



