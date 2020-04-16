from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from math import ceil, floor
from random import randint
from random import uniform
from random import seed
from time import time
import sys
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

filterwarnings('ignore')

class individual:
	def __init__(self, chrom):
		self.chrom = chrom
		self.bool_index = self.set_bool_index()
		self.atts = sum(self.bool_index)
		self.acc = self.set_acc()
		self.fitness = self.set_fitness()

	def set_bool_index(self):
		bool_index = [self.chrom[i] == 1 for i in range(chrom_size)]
		return bool_index

	def set_acc(self):
		ind_X_train = X_train.iloc[:, self.bool_index]
		cv_score = list()
		cv = 10
		folds = StratifiedKFold(n_splits = cv, shuffle = True)
		for train_index, test_index in folds.split(ind_X_train, y_train):
			x_Train, x_Val = ind_X_train.loc[train_index,:], ind_X_train.loc[test_index,:]
			y_Train, y_Val = y_train[train_index], y_train[test_index]
			clf.fit(x_Train, y_Train)
			score = accuracy_score(y_Val, clf.predict(x_Val))
			cv_score.append(score)
		mean_acc = sum(cv_score)/cv
		return mean_acc

	def set_fitness(self):
		if alg_option in fitness_options:
			fitness = fitness_funtion(self.acc, self.atts)
			if maximize_fitness == False:
				fitness = - fitness
		return fitness


def fitness_funtion(acc, atts):
	error_rate = 1 - acc
	fitness = a * error_rate + b * atts / chrom_size
	return fitness

def set_test_acc(columns):
	ind_X_train = original_X_train.loc[:, columns]
	ind_X_test = X_test.loc[:, columns]
	pred_test = clf.fit(ind_X_train, y_train).predict(ind_X_test)
	test_acc = accuracy_score(y_test, pred_test)
	return test_acc

def round_off(n, decimal_places = 4):
	m = 10 ** (decimal_places + 1)
	n = floor(n * m)/10
	f = floor(n)
	r = n - f
	if r >= 0.5:
		n = ceil(n)
	else:
		n = f
	m /= 10
	n /= m

	return n

def sort_population(pop, pop_size):
	fit_index = [[pop[i].fitness, i] for i in range(pop_size)]
	fit_index.sort(reverse = True)
	indexes = [fit_index[i][1] for i in range(pop_size)]
	sorted_pop = [pop[i] for i in indexes]
	return sorted_pop

def initial_population(n_individuals):
	pop = []
	for i in range(n_individuals):
		while(1):
			chrom = [randint(0, 1) for i in range(chrom_size)]
			if sum(chrom) != 0:
				break
		pop.append(individual(chrom))
	return pop

def Print(pop, pop_size, type_of_pop = None):
	if type_of_pop == 'best':
		print('Best individuals:')
	elif type_of_pop == 'final':
		print('Final individuals:')
	else:
		print('Results whithout reduction:')

	print('Atts    Acc      Fit')

	for i in range(pop_size):
		fitness = pop[i].fitness
		if maximize_fitness == False:
			fitness = - fitness
		print(f'{pop[i].atts:4}   {pop[i].acc:.4f}   {fitness:.4f}')
	print()

def tournament_selection():
	selected_ind_fitness = {}
	for i in range(n_selected):
		n = randint(0, pop_size - 1)
		selected_ind_fitness[pop[n].fitness] = n
	index_max_fitness = selected_ind_fitness[max(selected_ind_fitness.keys())]
	return pop[index_max_fitness]

def intermediate_population():
	int_pop = [tournament_selection() for i in range(int_pop_size)]
	return int_pop

def mutation(child_chrom):
	mutated_chrom = child_chrom
	while(1):
		for i in range(chrom_size):
			mutation_prob = uniform(0, 1)
			if mutation_prob < mutation_rate:
				if mutated_chrom[i] == 1:
					mutated_chrom[i] = 0
				else:
					mutated_chrom[i] = 1
		if sum(mutated_chrom) != 0:
			break
		else:
			mutated_chrom = child_chrom
	return individual(mutated_chrom)

def crossover(parent1, parent2):
	n = randint(0, chrom_size - 1)
	child_chrom = parent1.chrom[:n] + parent2.chrom[n:]
	return child_chrom

def new_generation():
	new_pop = []
	i = 0
	while len(new_pop) != pop_size:
		parent1 = int_pop[i]
		parent2 = int_pop[i+1]
		child_chrom = crossover(parent1, parent2)
		new_pop.append(mutation(child_chrom))
		i += 2
		if i == int_pop_size:
			i = 0
	new_pop = sort_population(new_pop, pop_size)
	new_pop.pop()
	return [pop[0]] + new_pop

def verify_AG_stagnation(best_ind1, best_ind2):
	global stagnant_gen
	global best_individual
	if best_ind1.chrom != best_ind2.chrom:
		best_individual = best_ind2
		stagnant_gen = 0
	else:
		stagnant_gen += 1

def attributes_frequency():
	att_freqs = []
	for i in range(chrom_size):
		att_freq = 0
		for j in range(n_best_inds):
			att_freq += best_individuals[j].chrom[i]
		att_freqs.append(att_freq)
	return att_freqs

def selected_attributes(att_rate, att_freqs):
	att_number = ceil(att_rate * n_best_inds)
	sel_attributes = []
	for i in range(chrom_size):
		if att_freqs[i] >= att_number:
			sel_attributes.append(1)
		else:
			sel_attributes.append(0)
	return sel_attributes

def generate_final_inds():
	final_inds = []
	att_freqs = attributes_frequency()
	for att_rate in att_rates:
		sel_attributes = selected_attributes(att_rate, att_freqs)
		if sum(sel_attributes) == 0:
			sel_attributes = [1] * chrom_size
		final_ind = individual(sel_attributes)
		final_inds.append(final_ind)
	final_inds = sort_population(final_inds, n_final_inds)
	return final_inds

def classifier(clf_option):
	if clf_option == 0:
		clf = GaussianNB()
	elif clf_option == 1:
		clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = None)
	elif clf_option == 2:
		clf = SVC(random_state = 0, max_iter = 50, gamma = 'auto')
	return clf

def split_X_y(file_name):
	dataset = pd.read_csv(file_name)
	data = dataset.iloc[:, :-1]
	target = dataset.iloc[:, -1]
	return data, target

def update_data():
	global X_train
	global column_names
	global chrom_size
	reduced_attributes = best_solution.bool_index
	X_train = X_train.iloc[:, reduced_attributes]
	column_names = column_names[reduced_attributes]
	chrom_size = sum(reduced_attributes)

def calculate_int_pop_size():
	int_pop_size = int(pop_size*0.7)
	if int_pop_size%2 != 0:
		int_pop_size += 1
	return int_pop_size

def replace_dot(results):
	n_values = len(results['Train accuracy'])
	for k in results.keys():
		for n in range(n_values):
			results[k][n] = str(results[k][n]).replace('.', ',')
	return results

def verify_round_stagnation():
	global stagnant_round
	global Round
	global best_round

	last_fit = rounds_results['Fitness'][Round - 1]
	current_fit = rounds_results['Fitness'][Round]

	if current_fit > last_fit:
		stagnant_round = 0
		return False
	elif current_fit < last_fit:
		best_round = Round - 1
		return True
	else:
		stagnant_round += 1
		if stagnant_round == max_stagnant_round:
			best_round = Round
			return True

def save_results(results, type_of_results = None):
	results = replace_dot(results)
	results_df = pd.DataFrame(results)

	if type_of_results == 'alg':
		file_name = f'../results/alg{alg_option}_clf{clf_option}_{dataset}_(fitness)_results.csv'
	elif type_of_results == 'round':
		file_name = f'../results/alg{alg_option}_clf{clf_option}_round{Round}_(fitness)_results.csv'

	results_df.to_csv(file_name, index = False, sep = ';')

def verify_typing_error(option, Dict, type_of_option):
	n_options = len(Dict)
	while option not in Dict.keys():
		print()
		print(f'Invalid {type_of_option} option. Please type again:\n', end = '')
		for i in range(n_options):
			print(f' {Dict[i]}: {i}')
		option = int(input())
	return option

def read_option(Dict, type_of_option):
	print(f'\nChoose an {type_of_option}:')

	n_options = len(Dict)

	for i in range(n_options):
		print(f' {Dict[i]}: {i}')
	option = int(input())

	return option

def set_options():
	if len(sys.argv) == 5:
		alg_option = int(sys.argv[1])
		clf_option = int(sys.argv[2])
		train_file_name = sys.argv[3]
		test_file_name = sys.argv[4]

		alg_option = verify_typing_error(alg_option, alg_option_dict, 'algorithm')
		clf_option = verify_typing_error(clf_option, clf_option_dict, 'classifier')

	else:
		alg_option = read_option(alg_option_dict, 'algorithm')
		alg_option = verify_typing_error(alg_option, alg_option_dict, 'algorithm')

		clf_option = read_option(clf_option_dict, 'classifier')
		clf_option = verify_typing_error(clf_option, clf_option_dict, 'classifier')

		train_file_name = input('Train file name: ')
		test_file_name = input('Test file name: ')

	print(f'\nAlgorithm option: {alg_option_dict[alg_option]}')
	print(f'Classifier option: {clf_option_dict[clf_option]}')
	print(f'Train file name: {train_file_name}')
	print(f'Test file name: {test_file_name}')

	dataset = train_file_name.split('/')[-1].split('(')[0]

	return alg_option, clf_option, train_file_name, test_file_name, dataset

def update_results(dict_results, best_solution, type_of_results):
	if type_of_results == 'round':
		ind = best_solution
		dict_results['Round'].append(Round)
		dict_results['Nº of attributes'].append(ind.atts)
		dict_results['Train accuracy'].append(ind.acc)
		dict_results['Time(min)'].append(Time)
		dict_results['Attributes'].append(column_names[ind.bool_index])
		dict_results['Fitness'].append(ind.fitness)
	else:
		columns = rounds_results['Attributes'][best_round]
		fitness = rounds_results['Fitness'][best_round]
		if maximize_fitness == False:
			fitness = - fitness
		dict_results['s'].append(s)
		dict_results['Last round'].append(Round)
		dict_results['Best round'].append(best_round)
		dict_results['Nº of attributes'].append(rounds_results['Nº of attributes'][best_round])
		dict_results['Train accuracy'].append(round_off(rounds_results['Train accuracy'][best_round]))
		dict_results['Test accuracy'].append(round_off(set_test_acc(columns)))
		dict_results['Time(min)'].append(round_off(total_time))
		dict_results['Attributes'].append([col for col in columns])
		dict_results['Fitness'].append(round_off(fitness))
	return dict_results



def generate_dict(keys):
	d = {keys[i]: list() for i in range(len(keys))}
	return d

def execution_0(dict_results, type_of_results):
	if type_of_results == 'alg':
		start = time()
	ind = individual([1] * chrom_size)
	if type_of_results == 'alg':
		Time = time() - start

	if type_of_results == 'alg':
		fitness = ind.fitness
		if maximize_fitness == False:
			fitness = - fitness
		test_acc = set_test_acc(column_names)
		dict_results['s'].append(0)
		dict_results['Last round'].append('-')
		dict_results['Best round'].append('-')
		dict_results['Nº of attributes'].append(ind.atts)
		dict_results['Train accuracy'].append(round_off(ind.acc))
		dict_results['Test accuracy'].append(round_off(test_acc))
		dict_results['Time(min)'].append(round_off(Time))
		dict_results['Attributes'].append([col for col in column_names])
		dict_results['Fitness'].append(round_off(fitness))
		save_results(dict_results, type_of_results = 'alg')
	else:
		dict_results['Round'].append('-')
		dict_results['Nº of attributes'].append(ind.atts)
		dict_results['Train accuracy'].append(ind.acc)
		dict_results['Time(min)'].append('-')
		dict_results['Attributes'].append([col for col in column_names])
		dict_results['Fitness'].append(ind.fitness)

	Print([ind], 1)
	return dict_results

def get_best_solution():
	if clf_option in final_options:
		best_solution = final_inds[0]
	else:
		best_solution = best_individuals[0]
	return best_solution




alg_option_dict = {0: 'Best',
				   1: 'Best(function)',
				   2: 'Final',
				   3: 'Final(function)'}

clf_option_dict = {0: 'GaussianNB',
				   1: 'DecisionTree',
				   2: 'SVM'}

alg_option, clf_option, train_file_name, test_file_name, dataset = set_options()

final_options = [2, 3]
fitness_options = [1, 3]

clf = classifier(clf_option)

#pop_sizes = [12, 20, 50]
pop_sizes = [12]
pop_size = int()
n_selected = 3
#mutation_rates = [0.02, 0.05, 0.10]
mutation_rates = [0.05]
mutation_rate = float()
max_stagnant_gen = 5
n_best_inds = 10
max_round = 10
att_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_final_inds = len(att_rates)
max_stagnant_round = 2
best_round = int()
sampling = 15
a = 0.99
b = 0.01
maximize_fitness = False




rounds_results_columns = [
	'Round',
	'Nº of attributes',
	'Train accuracy',
	'Fitness',
	'Time(min)',
	'Attributes'
]

alg_results_columns = [
	's',
	'Last round',
	'Best round',
	'Nº of attributes',
	'Train accuracy',
	'Fitness',
	'Test accuracy',
	'Time(min)',
	'Attributes'
]





for pop_size in pop_sizes:
	for mutation_rate in mutation_rates:

		alg_results = generate_dict(alg_results_columns)

		for s in range(sampling + 1):

			Times = list()
			rounds_results = generate_dict(rounds_results_columns)

			X_train, y_train = split_X_y(train_file_name)
			X_test, y_test = split_X_y(test_file_name)
			original_X_train = X_train

			column_names = X_train.columns
			chrom_size = len(column_names)
			int_pop_size = calculate_int_pop_size()

			stagnant_round = 0

			print(f'\ns: {s}')
			print(f'Population size: {pop_size}')
			print(f'Mutation rate: {mutation_rate}')
			print(f'Dataset: {dataset}\n')

			if s == 0:
				alg_results = execution_0(alg_results, 'alg')
				continue

			for Round in range(max_round + 1):
				print(f'--------------- Round {Round} ---------------')

				if Round == 0:
					rounds_results = execution_0(rounds_results, 'round')
					continue

				start = time()

				best_individuals = []

				for i in range(n_best_inds):
					seed()
					gen = 1
					stagnant_gen = 0

					pop = initial_population(pop_size)
					pop = sort_population(pop, pop_size)
					best_individual = pop[0]
			
					while(stagnant_gen <= max_stagnant_gen):
						int_pop = intermediate_population()
						gen += 1
						pop = new_generation()
						pop = sort_population(pop, pop_size)
						verify_AG_stagnation(best_individual, pop[0])

					best_individuals.append(pop[0])


				best_individuals = sort_population(best_individuals, n_best_inds)
				Print(best_individuals, n_best_inds, 'best')

				if alg_option in final_options:
					final_inds = generate_final_inds()
					final_inds = sort_population(final_inds, n_final_inds)
					Print(final_inds, n_final_inds, 'final')

				end = time()
				Time = (end - start) / 60

				best_solution = get_best_solution()
				rounds_results = update_results(rounds_results, best_solution, 'round')

				print(f'Time(minute): {round_off(Time):.4f}\n')
				Times.append(Time)

				update_data()

				if verify_round_stagnation():
					break

			total_time = sum(Times)
			print(f'Total time(minute): {total_time:.4f}\n')

			alg_results = update_results(alg_results, best_solution, 'alg')
			save_results(alg_results, 'alg')
