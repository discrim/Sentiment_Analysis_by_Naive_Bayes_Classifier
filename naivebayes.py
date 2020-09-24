import nltk
from nltk.tokenize import word_tokenize as wt
import pickle

# Description: Trains the naive Bayes classifier.
# Inputs: String for the file location of the training data (the "training" directory).
# Outputs: An object representing the trained model.
def train(training_path):
    print("Start train module")
    
    featuresets = 0
    try: # If train featuresets exist
        with open ('train_featuresets.dmp', 'rb') as fp:
            print("Train featuresets found!")
            featuresets = pickle.load(fp)
    except:
        print("No existing train featuresets. Make a new one.")
        
        # Data loading
        from os import listdir
        pospath = training_path + r"\pos"
        negpath = training_path + r"\neg"
        posfiles = listdir(pospath)
        negfiles = listdir(negpath)
        
        documents = []
        for fn in posfiles:
            ff = open(pospath + '/' + fn, encoding='windows-1252')
            ftok = [word.lower() for word in wt(ff.read())]
            documents.append((ftok, 'pos'))
        for fn in negfiles:
            ff = open(negpath + '/' + fn, encoding='windows-1252')
            ftok = [word.lower() for word in wt(ff.read())]
            documents.append((ftok, 'neg'))
        
        # Preprocessing - Vocabulary making
        vocab = []
        for doc in documents:
            for word in doc[0]:
                vocab.append(word)
        vocab = nltk.FreqDist(vocab)
        vocab = list(vocab.keys())
        
        # Preprocessing - Occurrence counting
        def countFeatures(body):
            features = {}
            for wd in vocab:
                features[wd] = body.count(wd)
            return features
        
        print("Start feature")
        featuresets = []
        count = 0
        for body, senti in documents:
            count += 1
            featuresets.append((countFeatures(body), senti))
            print("Doc " + str(count) + " done")
        
        print(featuresets[0])
        with open('train_featuresets.dmp', 'xb') as fp:
            pickle.dump(featuresets, fp)
        
    try: # If trained model exists
        with open('trained_model.dmp', 'rb') as fp:
            print("Trained model found!")
            trained_model = pickle.load(fp)
    except:
        # Model training
        print("No existing trained model. Make a new one.")
        trained_model = nltk.NaiveBayesClassifier.train(featuresets)
        with open('trained_model.dmp', 'xb') as fp:
            pickle.dump(trained_model, fp)
    
    print("Finish train module")
    return trained_model

# Description: Runs prediction of the trained naive Bayes classifier on the test set, and returns these predictions.
# Inputs: An object representing the trained model (whatever is returned by the above function), and a string for the file location of the test data (the "testing" directory).
# Outputs: An object representing the predictions of the trained model on the testing data, and an object representing the ground truth labels of the testing data.
def predict(trained_model, testing_path):
    print("Start predict module")
    try:
        with open('test_featuresets.dmp', 'rb') as fp:
            print("Test featuresets found!")
            documents = pickle.load(fp)
    except:
        print("No existing test featuresets. Make a new one.")
        
        from os import listdir
        pospath = testing_path + r"\pos"
        negpath = testing_path + r"\neg"
        posfiles = listdir(pospath)
        negfiles = listdir(negpath)
        
        documents = []
        for fn in posfiles:
            ff = open(pospath + '/' + fn, encoding='windows-1252')
            ftok = [word.lower() for word in wt(ff.read())]
            fset = set(ftok)
            tmpdict = {}
            for key in fset:
                tmpdict[key] = ftok.count(key)
            documents.append((tmpdict, 'pos'))
        for fn in negfiles:
            ff = open(negpath + '/' + fn, encoding='windows-1252')
            ftok = [word.lower() for word in wt(ff.read())]
            fset = set(ftok)
            tmpdict = {}
            for key in fset:
                tmpdict[key] = ftok.count(key)
            documents.append((tmpdict, 'neg'))
        
        with open('test_featuresets.dmp', 'xb') as fp:
            pickle.dump(documents, fp)
        
    model_predictions = []
    ground_truth = []
    count = 0
    for body, senti in documents:
        count += 1
        model_predictions.append(trained_model.classify(body))
        ground_truth.append(senti)
        print("Doc " + str(count) + " Done")
    
    print("Finish predict module")
    return model_predictions, ground_truth


# Description: Evaluates the accuracy of model predictions using the ground truth labels.
# Inputs: An object representing the predictions of the trained model, and an object representing the ground truth labels for the testing data.
# Outputs: Floating-point accuracy of the trained model on the test set.
def evaluate(model_predictions, ground_truth):
    print("Start evaluate module")
    right_count = 0
    wrong_count = 0
    for ii in range(len(model_predictions)):
        if model_predictions[ii] == ground_truth[ii]:
            right_count += 1
        else:
            wrong_count += 1
    accuracy = right_count / (right_count + wrong_count)
    print("Finish evaluate module")
    return accuracy


if __name__ == "__main__":
    TRAINING_PATH=r'D:\DevPath\F20_EECS595\HW1\Naive_Bayes\data\training'
    TESTING_PATH=r'D:\DevPath\F20_EECS595\HW1\Naive_Bayes\data\testing'

    trained_model = train(TRAINING_PATH)
    model_predictions, ground_truth = predict(trained_model, TESTING_PATH)
    accuracy = evaluate(model_predictions, ground_truth)
    print('Accuracy: %s' % str(accuracy))
