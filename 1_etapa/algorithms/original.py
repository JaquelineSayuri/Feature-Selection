from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from time import time
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sys


def split_X_y(dataset):
	data = dataset.iloc[:, :-1]
	target = dataset.iloc[:, -1]
	return data, target

def classifier(clf_option):
	if clf_option == 'naive bayes':
		clf = GaussianNB()
	elif clf_option == 'decision tree':
		clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = None)
	elif clf_option == 'SVM':
		clf = SVC(C = 0.001, gamma = 0.001, kernel = 'linear')
	return clf

def train_clf():
	global mean_train_scores
	global mean_test_scores
	global mean_times

	train_scores = list()
	test_scores = list()
	times = list()

	overfit = 0
	s = 30

	for i in range(s):

		clf = classifier(clf_option)
		
		begin = time()
		train_score = cross_val_score(clf, train_data, y_train, cv = 10, scoring = 'accuracy')
		train_score = train_score.mean()		
		end = time()

		clf.fit(train_data, y_train)
		y_test_pred = clf.predict(test_data)
		test_score = metrics.accuracy_score(y_test, y_test_pred)
		Time = end - begin
		if train_score - test_score >= 0.05:
			overfit += 1
		
		train_scores.append(train_score)
		test_scores.append(test_score)
		times.append(Time)


	mean_train_score = sum(train_scores)/s
	mean_test_score = sum(test_scores)/s
	mean_time = sum(times)/s

	print(f'Train acc: {mean_train_score}'.replace('.', ','))
	print(f'Test acc: {mean_test_score}'.replace('.', ','))
	print(f'Overfit: {overfit}/{s}')
	print(f'Time: {mean_time}'.replace('.', ','))
	print()

	mean_train_scores.append(mean_train_score)
	mean_test_scores.append(mean_test_score)
	mean_times.append(mean_time)


'''
# colon_cancer
train_file_names = [
	'../datasets/colon_cancer(train).csv',
	'../datasets/colon_cancer(train)stand.csv'
]

test_file_names = [
	'../datasets/colon_cancer(test).csv',
	'../datasets/colon_cancer(test)stand.csv'
]


# madelon
train_file_names = [
	'../datasets/madelon(train).csv',
	'../datasets/madelon(train)stand.csv'
]

test_file_names = [
	'../datasets/madelon(test).csv',
	'../datasets/madelon(test)stand.csv'
]


'''
# PCMAC
train_file_names = [
	'../datasets/PCMAC(train).csv',
	#'../datasets/PCMAC(train)stand.csv'
]

test_file_names = [
	'../datasets/PCMAC(test).csv',
	#'../datasets/PCMAC(test)stand.csv'
]
#'''

clfs = [
	#'naive bayes',
	#'decision tree',
	'SVM'
]


results = list()

if len(sys.argv) != 2:
	print_option = input('print option (y, n): ')
else:
	print_option = sys.argv[1]


for clf_option in clfs:
	print(clf_option)

	mean_train_scores = list()
	mean_test_scores = list()
	mean_times = list()

	for i in range(0, len(train_file_names)):
		print(train_file_names[i])
		print(test_file_names[i])

		train_data = pd.read_csv(train_file_names[i])
		test_data =  pd.read_csv(test_file_names[i])

		X_train, y_train = split_X_y(train_data)
		X_test, y_test = split_X_y(test_data)

		if print_option == 'y':
			print(train_data.head())
			print(test_data.head())

			print(train_data.shape)
			print(test_data.shape)

			print(y_train.head())
			print(y_test.head())

			print(y_train.value_counts())
			print(y_test.value_counts())

		print('Training the model...')

		train_clf()

	results.append(mean_train_scores)
	results.append(mean_test_scores)
	results.append(mean_times)

'''
for i in range(3 * len(clfs)):
	for j in range(len(train_file_names)):
		results[i][j] = str(results[i][j]).replace('.', ',')

results = pd.DataFrame(results)
print(results)

results.to_csv('../results/results.csv', header = False, index = False)
'''