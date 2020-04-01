import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def split_X_y(dataset):
	data = dataset.iloc[:, :-1]
	target = dataset.iloc[:, -1]
	return data, target


def normalize(X, minmax = False, standard = False):
	if minmax == True:
		scaler = MinMaxScaler()
	elif standard == True:
		scaler = StandardScaler()

	scaler.fit(X)
	scaled_X = scaler.transform(X)

	return scaled_X


def concatenate_X_y(X, y, columns):
	X_df = pd.DataFrame(X, columns = columns[:-1])
	y_series = pd.Series(y, name = columns[-1])
	data = pd.concat([X_df, y_series], axis = 1, ignore_index = True)

	return data



while(1):

	suffixes = list()

	file_name = input('File name: ')
	data = pd.read_csv(file_name)
	#data = data.iloc[:, 1:]


	columns = data.columns
	X, y = split_X_y(data)
	print(columns)
	print(X)
	print(y)
	print(y.value_counts())

	
	le_option = input('Label encoder [y/n]: ')
	if le_option == 'y':
		le = LabelEncoder()
		le.fit(y)
		y = le.transform(y)
		y = pd.Series(y, name = columns[-1])
	print(y)

	
	rus_option = input('Random under sampler [y/n]: ')
	if rus_option == 'y':
		suffixes.append('rus')
		rus = RandomUnderSampler(random_state = 0)
		X, y = rus.fit_resample(X, y)
	print(X)
	print(y)
	

	minmax_option = input(' MinMax Scaler [y/n]: ')
	if minmax_option == 'y':
		suffixes.append('minmax')
		X = normalize(X, minmax = True)
		X = pd.DataFrame(X, columns = columns[:-1])
	print(X)


	stand_option = input(' Standard Scaler [y/n]: ')
	if stand_option == 'y':
		suffixes.append('stand')
		X = normalize(X, standard = True)
		X = pd.DataFrame(X, columns = columns[:-1])
	print(X)


	split_option = input(' Split data into train data and test data [y/n]: ')
	if split_option == 'y':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
		print(X_train)
		print(y_train)
		print(X_test)
		print(y_test)

		train_data = concatenate_X_y(X_train, y_train, columns)
		test_data = concatenate_X_y(X_test, y_test, columns)
		print(train_data)
		print(test_data)

		train_file_name = file_name[:-4] + '(train)_' + '_'.join(suffixes) + '.csv'
		test_file_name = file_name[:-4] + '(test)_' + '_'.join(suffixes) + '.csv'

		train_data.to_csv(train_file_name, index = False, header = columns)
		test_data.to_csv(test_file_name, index = False, header = columns)
	else:
		data = concatenate_X_y(X, y, columns)
		print(data)
		new_file_name = file_name[:-4] + '_' + '_'.join(suffixes) + '.csv'
		data.to_csv(new_file_name, index = False, header = columns)