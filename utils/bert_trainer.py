# Created by otrad at 04/02/2022
import os
import re
import string
import numpy as np
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


class BertTrainer:

    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.MAX_LENGTH = 200
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME,
                                                add_special_tokens=True,
                                                max_length=self.MAX_LENGTH,
                                                pad_to_max_length=True)

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

    def create_model(self):
        """
        Create model proper
        :return:
        """
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

        self.early_stopping = EarlyStopping(patience=3,  # Stop after 3 epochs of no improvement
                                       monitor='val_loss',  # Look at validation_loss
                                       min_delta=0,  # After 0 change
                                       mode='min',  # Stop when quantity has stopped decreasing
                                       restore_best_weights=False,  # Don't Restore the best weights
                                       verbose=1)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Look at validation loss
                                      min_lr=0.000001,  # Lower bound of learning rate
                                      patience=1,  # Reduce after 1 with little change
                                      mode='min',  # Stop when quantity has stopped decreasing
                                      factor=0.1,  # Reduce by a factor of 1/10
                                      min_delta=0.01,  # Minimumn change needed
                                      verbose=1)

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, x_train_pre, y_train, x_validation_pre, y_validation):
        """

        :param x_train_pre:
        :param y_train:
        :param x_validation_pre:
        :param y_validation:
        :return:
        """
        x_train = self.tokenize(x_train_pre)
        x_validation = self.tokenize(x_validation_pre)
        le = LabelEncoder()
        y_train_transformed = le.fit_transform(y_train)
        y_validation_transformed = le.fit_transform(y_validation)

        history = self.model.fit(x_train, y_train_transformed,
                            epochs=10,
                            batch_size=16,
                            validation_data=(x_validation, y_validation_transformed),
                            callbacks=[self.model_checkpoint, self.early_stopping, self.reduce_lr])
