This is a draft for a classifier to evaluate exams.
The labels have the following distribution:

correct_but_incomplete 210
correct 367
incorrect 237
contradictory 84

Initially no balance was done.

Two basic approaches are tested.
Check 
    main_basic.py
and
    main_transformer_training.py

The first one still has results well below acceptable precision and recall
and the second one was not even completed.

Below results for the naive (completed) first approach:

Prediction: 0.58%

                        precision    recall  f1-score   support

correct_but_incomplete       0.50      0.43      0.46         7
               correct       0.68      0.68      0.68        37
             incorrect       0.52      0.50      0.51        24
         contradictory       0.54      0.59      0.57        22

              accuracy                           0.59        90
             macro avg       0.56      0.55      0.55        90
          weighted avg       0.59      0.59      0.59        90



The first thing one needs to do is to use the reference answers
(a possibility is for a word2vec model) but making sure that does
not create more unbalance.
One possibility is using highly similar unique tokens (types) from
a w2v (gensim or the like) model to produce more examples for 
underrepresented classes. In that way we could actually use the 
 references for more of the correct class.
 
The other is a distill dilbert, which was not completed.

No unit tests were made yet