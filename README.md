# Sentiment_Analysis_by_Naive_Bayes_Classifier
## Logistics
- Affiliation: University of Michigan - Ann Arbor
- Term: 2020 Fall
- Course: EECS 595 Natural Language Processing
- Note: HW1
## Usage
```python
import naivebayes

TRAINING_PATH=r'path\to\train_data'
TESTING_PATH=r'path\to\test_data'

trained_model = naivebayes.train(TRAINING_PATH)
model_predictions, ground_truth = naivebayes.predict(trained_model, TESTING_PATH)
accuracy = naivebayes.evaluate(model_predictions, ground_truth)
print('Accuracy: %s' % str(accuracy))
```
```
DATE is DATE. I have DOLLAR_AMOUNT in my pocket. I am going to buy a nice jersey from WEB_ADDRESS with this money.
```
## Problem Statement
This is a simple Naive Bayes Classifier for sentiment analysis of movie reviews. I used the standard dataset (`from nltk.corpus import movie_reviews`) to analyze. Given multiple movie reviews and their class - positive or negative - this classifier learns from those and become able to determine a document's class even it never saw that document before.
## Short Description
The directory for dataset should be as follows:
```
┬train
│└pos
│└neg
└test
 └pos
 └neg

```
There are three modules (functions): `train`, `predict`, `evaluate`.
- `train`: Get the training set and train the Naive Bayes Classifier.
- `predict`: Get the test set and predict whether the given review is positive or negative.
- `evaluate`: Calculate the accuracy 'correct cases / (correct cases + wrong cases)'.
Note that this code DOES NOT USE the movie review imported from nltk; I used the identical ones from different storage.
## Feature Information
My features have keys and values. Keys are words that appear in one document, and its corresponding values are the number of occurrence of that key in the document. For example, a feature for a document looks like {'this': 3, 'apple': 10, ... }. I first tried to compare all the tokens of each document with the whole vocabulary so the feature extraction took forever. After I changed the plan to let the dictionary of each document contain words only in that document, the time became reasonable; 9-minute train, 2-second predict, and instant evaluate (with 'print' function calls after extracting features from each document). However, I still think that there is a way to make the total runtime less that 1 minute. Once I know the exact method, I will update the code.
## Result
The accuracy is 70.71%.
