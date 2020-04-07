from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from math import ceil, floor
from random import randint
from random import uniform
from random import seed
from time import time
from sklearn.model_selection import cross_val_score
import sys
from warnings import filterwarnings
from sklearn.model_selection import train_test_split

filterwarnings('ignore')

class individual:
	def __init__(self, chrom):
		self.chrom = chrom
		self.bool_index = self.set_bool_index()
		self.atts = sum(self.bool_index)
		self.acc = self.set_acc()
		self.fitness = self.set_fitness()
		self.val_acc = None

	def set_bool_index(self):
		bool_index = [self.chrom[i] == 1 for i in range(chrom_size)]
		return bool_index

	def set_acc(self):
		ind_X_train = X_train.iloc[:, self.bool_index]
		accs = cross_val_score(clf, ind_X_train, y_train, cv = 10)
		mean_acc = accs.mean()
		return round_off(mean_acc)

	def set_fitness(self):
		if alg_option in fitness_options:
			fitness = 100 * self.acc - self.atts / 10
		else:
			fitness = self.acc
		return round_off(fitness)

	def set_val_acc(self):
		ind_X_train = X_train.iloc[:, self.bool_index]
		ind_X_val = X_val.iloc[:, self.bool_index]
		pred_val = clf.fit(ind_X_train, y_train).predict(ind_X_val)
		acc = (y_val == pred_val).sum() / len(y_val)
		self.val_acc = round_off(acc)


def set_test_acc(columns):
	ind_X_train = X.loc[:, columns]
	ind_X_test = X_test.loc[:, columns]

	pred_test = clf.fit(ind_X_train, y).predict(ind_X_test)
	acc = (y_test == pred_test).sum() / len(y_test)
	return round_off(acc)

def round_off(n, decimal_places = 2):
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
	acc_att_index = [[pop[i].fitness, pop[i].atts, i] for i in range(pop_size)]
	acc_att_index.sort(reverse = True)

	first_acc = acc_att_index[0][0]
	n_same_accs = 0
	for i in range(pop_size):
		if acc_att_index[i][0] == first_acc:
			n_same_accs += 1
		else:
			break

	if n_same_accs > 1:
		att_index = [[acc_att_index[i][1], acc_att_index[i][2]] for i in range(n_same_accs)]
		att_index.sort()
		indexes = [att_index[i][1] for i in range(n_same_accs)] + [acc_att_index[i][2] for i in range(n_same_accs, pop_size)]		
	else:
		indexes = [acc_att_index[i][2] for i in range(pop_size)]

	sorted_pop = [pop[i] for i in indexes]
	return sorted_pop

def initial_population():
	pop = []
	for i in range(pop_size):
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
		print('Results without reduction:')

	if alg_option not in fitness_options:
		print('Atts    Acc   Val_acc    Dif')
	else:
		print('Atts    Acc   Val_acc    Dif      Fit')

	for i in range(pop_size):
		dif = round_off(pop[i].acc - pop[i].val_acc)
		if dif < 0:
			dif = f'{dif:.4f}'
		else:
			dif = f'+{dif:.4f}'
		print(f'{pop[i].atts:3}   {pop[i].acc:.4f}   {pop[i].val_acc:.4f}   {dif}', end = '')
		if alg_option in fitness_options:
			print(f'   {pop[i].fitness:.2f}')
		else:
			print()
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
	if best_ind1.fitness < best_ind2.fitness:
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

def set_pop_val_accs(pop, pop_size):
	for i in range(pop_size):
		pop[i].set_val_acc()
	return pop

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
	global X_val
	global column_names
	global chrom_size
	X_train = X_train.iloc[:, reduced_attributes]
	X_val = X_val.iloc[:, reduced_attributes]
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

def update_rounds_results(results, pop, pop_size, overfit = None):
	ind = pop[0]

	if overfit == False:
		for i in range(pop_size):
			dif = round_off(pop[i].acc - pop[i].val_acc)
			if dif < 0.05:
				ind = pop[i]
				break

	results['Round'].append(Round)
	results['Nº of attributes'].append(ind.atts)
	results['Train accuracy'].append(ind.acc)
	results['Validation accuracy'].append(ind.val_acc)
	results['Difference'].append(round_off(ind.acc - ind.val_acc))
	results['Time(min)'].append(round_off(Time))
	results['Attributes'].append(column_names[ind.bool_index])

	if alg_option in fitness_options:
		results['Fitness'].append(round_off(ind.fitness))
	
	return results

def mean_acc(pop, pop_size):
	total_acc = 0
	for i in range(pop_size):
		total_acc += pop[i].acc
	return total_acc / pop_size

def verify_round_stagnation():
	global stagnant_round

	if Round == 1:
		return False
	else:
		last_acc = rounds_results['Train accuracy'][Round - 2]
		current_acc = rounds_results['Train accuracy'][Round - 1]
		last_acc = float(last_acc)
		current_acc = float(current_acc)

		if current_acc > last_acc:
			stagnant_round = 0
			return False
		elif current_acc < last_acc:
			return True
		else:
			last_atts = rounds_results['Nº of attributes'][Round - 2]
			current_atts = rounds_results['Nº of attributes'][Round - 1]
			if current_atts == last_atts:
				stagnant_round += 1
				if stagnant_round == max_stagnant_round:
					return True
			else:
				stagnant_round = 0
				return False

def save_results(results, type_of_results = None):
	results = replace_dot(results)
	results_df = pd.DataFrame(results)

	if type_of_results == 'alg':
		file_name = f'../results/alg{alg_option}_clf{clf_option}_{dataset}_results.csv'
	elif type_of_results == 'round':
		file_name = f'../results/alg{alg_option}_clf{clf_option}_round{Round}_results.csv'

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

def update_alg_results(results):

	last_round = rounds_results['Round'][-1]
	best_round = last_round

	for i in range(last_round, 0, -1):
		if results['Difference'][i - 1] < 0.05:
			best_round = i
			break

	columns = results['Attributes'][best_round - 1]

	alg_results['s'].append(s)
	alg_results['Last round'].append(last_round)
	alg_results['Best round'].append(results['Round'][best_round - 1])
	alg_results['Nº of attributes'].append(results['Nº of attributes'][best_round - 1])
	alg_results['Train accuracy'].append(results['Train accuracy'][best_round - 1])
	alg_results['Validation accuracy'].append(results['Validation accuracy'][best_round - 1])
	alg_results['Test accuracy'].append(set_test_acc(columns))
	alg_results['Difference'].append(results['Difference'][best_round - 1])
	alg_results['Time(min)'].append(total_time)
	alg_results['Attributes'].append([col for col in columns])

	if alg_option in fitness_options:
		alg_results['Fitness'].append(results['Fitness'][best_round - 1])

	return alg_results

def generate_dict(keys):
	d = {keys[i]: list() for i in range(len(keys))}
	return d

def execute_s_0():
	ind = individual([1] * chrom_size)
	ind.set_val_acc()
	test_acc = set_test_acc(column_names)
	dif = round_off(ind.acc - ind.val_acc)

	alg_results['s'].append(0)
	alg_results['Last round'].append('-')
	alg_results['Best round'].append('-')
	alg_results['Nº of attributes'].append(ind.atts)
	alg_results['Train accuracy'].append(ind.acc)
	alg_results['Validation accuracy'].append(ind.val_acc)
	alg_results['Test accuracy'].append(test_acc)
	alg_results['Difference'].append(dif)
	alg_results['Time(min)'].append('-')
	alg_results['Attributes'].append([col for col in column_names])

	if alg_option in fitness_options:
		fitness = 100 * ind.acc - ind.atts / 10
		alg_results['Fitness'].append(fitness)

	Print([ind], 1)

	save_results(alg_results, type_of_results = 'alg')





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
#pop_sizes = [20]
pop_sizes = [12]
pop_size = int()

n_selected = 3

#mutation_rates = [0.02, 0.05, 0.10]
mutation_rates = [0.05]
mutation_rate = float()

#max_stagnant_gen = 30
max_stagnant_gen = 2
#n_best_inds = 10
n_best_inds = 4
#max_round = 15
max_round = 3
att_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_final_inds = len(att_rates)

#max_stagnant_round = 2
max_stagnant_round = 1
#sampling = 15
sampling = 5





rounds_results_columns = [
	'Round',
	'Nº of attributes',
	'Train accuracy',
	'Validation accuracy',
	'Difference',
	'Time(min)',
	'Attributes'
]

alg_results_columns = [
	's',
	'Last round',
	'Best round',
	'Nº of attributes',
	'Train accuracy',
	'Validation accuracy',
	'Test accuracy',
	'Difference',
	'Time(min)',
	'Attributes'
]

if alg_option in fitness_options:
	rounds_results_columns.append('Fitness')
	alg_results_columns.append('Fitness')





for pop_size in pop_sizes:
	for mutation_rate in mutation_rates:

		alg_results = generate_dict(alg_results_columns)

		for s in range(sampling + 1):

			Times = list()
			rounds_results = generate_dict(rounds_results_columns)
			rounds_results_without_overfit = generate_dict(rounds_results_columns)

			X, y = split_X_y(train_file_name)
			X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3)
			X_test, y_test = split_X_y(test_file_name)

			column_names = X.columns
			chrom_size = len(column_names)
			int_pop_size = calculate_int_pop_size()

			stagnant_round = 0

			print(f'\ns: {s}')
			print(f'Population size: {pop_size}')
			print(f'Mutation rate: {mutation_rate}')
			print(f'Dataset: {dataset}\n')

			if s == 0:
				execute_s_0()
				continue

			for Round in range(1, max_round + 1):
				print(f'--------------- Round {Round} ---------------')
				start = time()

				best_individuals = []

				for i in range(n_best_inds):
					seed()
					gen = 1
					stagnant_gen = 0

					pop = initial_population()
					pop = sort_population(pop, pop_size)
					best_individual = pop[0]
			
					while(stagnant_gen <= max_stagnant_gen):
						int_pop = intermediate_population()
						gen += 1
						pop = new_generation()
						pop = sort_population(pop, pop_size)
						verify_AG_stagnation(best_individual, pop[0])

					best_individuals.append(pop[0])


				best_individuals = set_pop_val_accs(best_individuals, n_best_inds)
				best_individuals = sort_population(best_individuals, n_best_inds)
				Print(best_individuals, n_best_inds, type_of_pop = 'best')

				if alg_option in final_options:
					final_inds = generate_final_inds()
					final_inds = set_pop_val_accs(final_inds, n_final_inds)
					final_inds = sort_population(final_inds, n_final_inds)

					reduced_attributes = final_inds[0].bool_index

					end = time()
					Time = (end - start) / 60

					Print(final_inds, n_final_inds, type_of_pop = 'final')
					rounds_results = update_rounds_results(rounds_results, final_inds, n_final_inds, overfit = True)
					rounds_results_without_overfit = update_rounds_results(rounds_results_without_overfit, final_inds, n_final_inds, overfit = False)
				else:
					reduced_attributes = best_individuals[0].bool_index

					end = time()
					Time = (end - start) / 60

					rounds_results = update_rounds_results(rounds_results, best_individuals, n_best_inds, overfit = True)
					rounds_results_without_overfit = update_rounds_results(rounds_results_without_overfit, best_individuals, n_best_inds, overfit = False)

					
				print(f'Time(minute): {round_off(Time):.4f}\n')
				Times.append(Time)

				update_data()

				if verify_round_stagnation():
					break

			total_time = round_off(sum(Times))
			print(f'Total time(minute): {total_time:.4f}\n')

			alg_results = update_alg_results(rounds_results_without_overfit)

			save_results(alg_results, type_of_results = 'alg')
