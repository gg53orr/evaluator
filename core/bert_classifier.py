# Created by otrad at 04/02/2022
import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import warnings
from transformers import logging as hf_logging

MODEL_NAME = 'distilbert-base-cased'
final_labels = ['incorrect','contradictory','correct_but_incomplete','correct']


def print_cf1(y_test, y_hat):
    """

    :param y_test:
    :param y_hat:
    :return:
    """
    cm = confusion_matrix(y_test, y_hat)
    sns.set(font_scale = 1.4, color_codes=True, palette="deep")
    sns.heatmap(pd.DataFrame(cm, index=final_labels,columns=[0,1,2,3,4]),
                annot = True,
                annot_kws = {"size":16},
                fmt="d",
                cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.xticks([0,1,2,3,4], final_labels, rotation=45)
    plt.ylabel("True Value")
    plt.show()


class BertTrainer:

    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.MAX_LENGTH = 200
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME,
                                                add_special_tokens=True,
                                                max_length=self.MAX_LENGTH,
                                                pad_to_max_length=True)
        config = DistilBertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True, output_attentions=True)
        DistilBERT = TFDistilBertModel.from_pretrained(MODEL_NAME, config=config)

        input_ids_in = tf.keras.layers.Input(shape=(self.MAX_LENGTH,),
                                             name='input_token', dtype='int32')
        input_masks_in = tf.keras.layers.Input(shape=(self.MAX_LENGTH,),
                                               name='masked_token', dtype='int32')

        embedding_layer = DistilBERT(input_ids=input_ids_in, attention_mask=input_masks_in)[0]
        X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embedding_layer)
        X = tf.keras.layers.GlobalMaxPool1D()(X)
        X = tf.keras.layers.Dense(64, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(5, activation='softmax')(X)
        self.model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)

        for layer in self.model.layers[:3]:
            layer.trainable = False

        self.model.summary()
        output_dir = './model1_outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.model_checkpoint = ModelCheckpoint(filepath=output_dir + '/weights.{epoch:02d}.hdf5',
                                           save_weights_only=True)

        self.early_stopping = EarlyStopping(patience=3,
                                       monitor='val_loss',
                                       min_delta=0,
                                       mode='min',
                                       restore_best_weights=False,
                                       verbose=1)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      min_lr=0.000001,
                                      patience=1,
                                      mode='min',
                                      factor=0.1,
                                      min_delta=0.01,
                                      verbose=1)

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def tokenize(self, sentences):
        """
        Tokenize all sentences
        :param sentences:
        :return:
        """
        input_ids, input_masks, input_segments = [], [], []
        for sentence in tqdm(sentences):
            inputs = self.tokenizer.encode_plus(sentence,
                                           add_special_tokens=True,
                                           max_length=self.MAX_LENGTH,
                                           pad_to_max_length=True,
                                           return_attention_mask=True,
                                           return_token_type_ids=True,
                                           truncation=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

    def train(self, x_train_pre, y_train, x_validation_pre, y_validation, epochs):
        """

        :param x_train_pre:
        :param y_train:
        :param x_validation_pre:
        :param y_validation:
        :param epochs:
        :return:
        """
        x_train = self.tokenize(x_train_pre)
        x_validation = self.tokenize(x_validation_pre)
        le = LabelEncoder()
        y_train_transformed = le.fit_transform(y_train)
        y_validation_transformed = le.fit_transform(y_validation)
        print("Start fitting:")
        self.history = self.model.fit(x_train, y_train_transformed,
                            epochs=epochs,
                            batch_size=16,
                            validation_data=(x_validation, y_validation_transformed),
                            callbacks=[self.model_checkpoint, self.early_stopping, self.reduce_lr])
        print(self.history)
        print("Finished fitting")
        #self.plot_history(self.history)

    def train_test(self, x_train_pre, y_train, x_validation_pre, y_validation, x_test, y_test, epochs):
        """

        :param x_train_pre:
        :param y_train:
        :param x_validation_pre:
        :param y_validation:
        :param epochs:
        :return:
        """
        x_train = self.tokenize(x_train_pre)
        x_validation = self.tokenize(x_validation_pre)
        le = LabelEncoder()
        y_train_transformed = le.fit_transform(y_train)
        y_validation_transformed = le.fit_transform(y_validation)
        print("Start fitting:")
        self.history = self.model.fit(x_train, y_train_transformed,
                            epochs=epochs,
                            batch_size=16,
                            validation_data=(x_validation, y_validation_transformed),
                            callbacks=[self.model_checkpoint, self.early_stopping, self.reduce_lr])
        print(self.history)
        epoch_num = self.get_max_val_acc_epoch(self.history)
        print("Epoch num:", epoch_num)
        self.model.load_weights(self.output_dir + "/weights." + epoch_num + ".hdf5")
        y_test_probs = self.model.predict(x_test)
        # Turn probabilities into integer prediction
        y_hat = []
        for prob in y_test_probs:
            y_hat.append(np.argmax(prob))
        print("Accuracy:", accuracy_score(y_test, y_hat))
        print_cf1(y_test, y_hat)
        print(classification_report(y_test, y_hat, target_names=final_labels))

    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        print("Lowest Validation Loss: epoch {}".format(np.argmin(val_loss) + 1))
        print("Highest Validation Accuracy: epoch {}".format(np.argmax(val_acc) + 1))

    def get_min_val_loss_epoch(self, history):
        return "0" + str(np.argmin(history.history['val_loss']) + 1)

    @staticmethod
    def get_max_val_acc_epoch(history):
        return "0" + str(np.argmax(history.history['val_accuracy']) + 1)

    def test(self, x_test, y_test):
        output_dir = './model1_outputs'
        epoch_num = self.get_max_val_acc_epoch(self.history)
        print("Epoch num:", epoch_num)
        self.model.load_weights(output_dir + "/weights." + epoch_num + ".hdf5")

        y_test_probs = self.model.predict(x_test)
        # Turn probabilities into integer prediction
        y_hat = []
        for prob in y_test_probs:
            y_hat.append(np.argmax(prob))
        print("Accuracy:", accuracy_score(y_test, y_hat))
        print_cf1(y_test, y_hat)
        print(classification_report(y_test, y_hat, target_names=final_labels))
