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
Total training: 718
Prediction: 0.55%
                        precision    recall  f1-score   support

correct_but_incomplete       0.38      0.43      0.40         7
               correct       0.63      0.65      0.64        37
             incorrect       0.48      0.42      0.44        24
         contradictory       0.57      0.59      0.58        22

              accuracy                           0.56        90
             macro avg       0.51      0.52      0.52        90
          weighted avg       0.55      0.56      0.55        90


The first thing one needs to do is to use the reference answers
(a possibility is for a word2vec model) but making sure that does
not create more unbalance.
Another possibility is to produce a better normalisation.
Another is to increase the training. 

The other is a distill dilbert, which was not completed.

No unit tests were made yet