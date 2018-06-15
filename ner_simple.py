from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer,confusion_matrix
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer,FeatureHasher
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,classification_report
from sklearn.pipeline import Pipeline
import string

"""
Load the training/testing data. 
input: conll format data, but with only 2 tab separated colums - words and NEtags.
output: A list where each item is 2 lists.  sentence as a list of tokens, NER tags as a list for each token.
"""
def load__data_conll(file_path):
    myoutput,words,tags = [],[],[]
    fh = open(file_path)
    for line in fh:
        line = line.strip()
        if "\t" not in line:
            #Sentence ended.
            myoutput.append([words,tags])
            words,tags = [],[]
        else:
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
    fh.close()
    return myoutput


"""
Get features for all words in the sentence
input: sentence as a list of tokens.
output: list of dictionaries. each dict represents features for that word.
"""
def sent2feats(sentence):
    feats = []
    sen_tags = pos_tag(sentence) #This format is specific to this tagger!
    for i in range(0,len(sentence)):
        word = sentence[i]
        wordfeats = {}
       #word features: word, prev 2 words, next 2 words in the sentence.
        wordfeats['word'] = word
        if i == 0:
            wordfeats["prevWord"] = wordfeats["prevSecondWord"] = "<S>"
        elif i==1:
            wordfeats["prevWord"] = sentence[0]
            wordfeats["prevSecondWord"] = "</S>"
        else:
            wordfeats["prevWord"] = sentence[i-1]
            wordfeats["prevSecondWord"] = sentence[i-2]
        #next two words as features
        if i == len(sentence)-2:
            wordfeats["nextWord"] = sentence[i+1]
            wordfeats["nextNextWord"] = "</S>"
        elif i==len(sentence)-1:
            wordfeats["nextWord"] = "</S>"
            wordfeats["nextNextWord"] = "</S>"
        else:
            wordfeats["nextWord"] = sentence[i+1]
            wordfeats["nextNextWord"] = sentence[i+2]
        
        #POS tag features: current tag, previous and next 2 tags.
        wordfeats['tag'] = sen_tags[i][1]
        if i == 0:
            wordfeats["prevTag"] = wordfeats["prevSecondTag"] = "<S>"
        elif i == 1:
            wordfeats["prevTag"] = sen_tags[0][1]
            wordfeats["prevSecondTag"] = "</S>"
        else:
            wordfeats["prevTag"] = sen_tags[i - 1][1]

            wordfeats["prevSecondTag"] = sen_tags[i - 2][1]
            # next two words as features
        if i == len(sentence) - 2:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
            wordfeats["nextNextTag"] = "</S>"
        elif i == len(sentence) - 1:
            wordfeats["nextTag"] = "</S>"
            wordfeats["nextNextTag"] = "</S>"
        else:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
            wordfeats["nextNextTag"] = sen_tags[i + 2][1]
        #That is it! You can add whatever you want!
        feats.append(wordfeats)
    return feats

#Extract features from the conll data, after loading it.
def get_feats_conll(conll_data):
    feats = []
    labels = []
    for sentence in conll_data:
        feats.append(sent2feats(sentence[0]))
        labels.append(sentence[1])
    return feats, labels

#Get features for a non-sequence model
def get_feats_conll_nonseq(conll_data):
    feats = []
    labels = []
    for sentence in conll_data:
        feats.extend(sent2feats(sentence[0]))
        labels.extend(sentence[1])
    return feats, labels

#Train a non-sequence model
def train_nonseq(train_data,train_labels,dev_data,dev_labels,model):
    text_clf = Pipeline([('vect', DictVectorizer()), ('clf', model)])
    text_clf.fit(train_data, train_labels)
    preds = text_clf.predict(dev_data)
    print(f1_score(dev_labels,preds,average="weighted"))
    labels = ["O","B-LOC","I-LOC","B-MISC","I-MISC","B-ORG","I-ORG","B-PER","I-PER"]
    print(print_cm(confusion_matrix(dev_labels, preds),labels=labels))
    print(classification_report(preds,dev_labels,labels=labels))

#Train a sequence model
def train_seq(X_train,Y_train,X_dev,Y_dev):
   # crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50, all_possible_states=True)
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=50)#, all_possible_states=True)
    #Just to fit on training data
    crf.fit(X_train, Y_train)
    labels = list(crf.classes_)
    #testing:
    y_pred = crf.predict(X_dev)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_f1_score(Y_dev, y_pred,average='weighted', labels=labels))
    print(metrics.flat_classification_report(Y_dev, y_pred, labels=sorted_labels, digits=3))
    print(metrics.sequence_accuracy_score(Y_dev, y_pred))
    get_confusion_matrix(Y_dev, y_pred,labels=sorted_labels)

#python-crfsuite does not have a confusion matrix function, so writing it using sklearn's confusion matrix and print_cm from github
def get_confusion_matrix(y_true,y_pred,labels):
    trues,preds = [], []
    for yseq_true, yseq_pred in zip(y_true, y_pred):
        trues.extend(yseq_true)
        preds.extend(yseq_pred)
    print_cm(confusion_matrix(trues,preds,labels),labels)

#source for this function: https://gist.github.com/zachguo/10296432
def print_cm(cm, labels):
    print("\n")
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        sum = 0
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            sum =  sum + int(cell)
            print(cell, end=" ")
        print(sum) #Prints the total number of instances per cat at the end.


def main():
    train_path = 'data/conll2003/en/ner/train.txt'
    test_path = 'data/conll2003/en/ner/test.txt'
    conll_train = load__data_conll(train_path)
    conll_dev = load__data_conll(test_path)
    
    print("Training a regular, non-sequence, classification model, with Random Forests")
    feats, labels = get_feats_conll_nonseq(conll_train)
    devfeats, devlabels = get_feats_conll_nonseq(conll_dev)
    train_nonseq(feats,labels,devfeats,devlabels,RandomForestClassifier())
    print("Done with it")

    print("Training a Sequence classification model with CRF")
    feats, labels = get_feats_conll(conll_train)
    devfeats, devlabels = get_feats_conll(conll_dev)
    train_seq(feats, labels, devfeats, devlabels)
    print("Done with sequence model")

if __name__=="__main__":
    main()

