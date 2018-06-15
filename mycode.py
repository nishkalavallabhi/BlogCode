from anago.utils import load_data_and_labels
import anago

x_train, y_train = load_data_and_labels('data/conll2003/en/ner/train.txt')
x_test, y_test = load_data_and_labels('data/conll2003/en/ner/test.txt')
x_dev, y_dev = load_data_and_labels('data/conll2003/en/ner/valid.txt')
model = anago.Sequence()
model.fit(x_train, y_train, x_dev, y_dev, epochs=15)
model.score(x_test,y_test)

