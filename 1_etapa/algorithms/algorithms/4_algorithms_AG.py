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

filterwarnings('ignore')

class individual:
	def __init__(self, chrom):
		self.chrom = chrom
		self.bool_index = self.set_bool_index()
		self.atts = sum(self.bool_index)
		self.acc = self.set_acc()
		self.fitness = self.set_fitness()
		self.test_acc = None

	def set_bool_index(self):
		bool_index = [self.chrom[i] == 1 for i in range(chrom_size)]
		return bool_index

	def set_acc(self):
		individual_dataset = train_data.iloc[:, self.bool_index]
		accs = cross_val_score(clf, individual_dataset, train_target, cv = 10)
		mean_acc = accs.mean()
		return round_off(mean_acc)

	def set_fitness(self):
		if alg_option in fitness_options:
			fitness = 100 * self.acc - self.atts / 10
		else:
			fitness = self.acc
		return round_off(fitness)

	def set_test_acc(self):
		ind_train_data = train_data.iloc[:, self.bool_index]
		ind_test_data = test_data.iloc[:, self.bool_index]
		test_pred = clf.fit(ind_train_data, train_target).predict(ind_test_data)
		acc = (test_target == test_pred).sum() / len(test_target)
		self.test_acc = round_off(acc)


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

	if alg_option not in fitness_options:
		print('Atts    Acc   Test_acc    Dif')
	else:
		print('Atts    Acc   Test_acc    Dif      Fit')

	for i in range(pop_size):
		dif = round_off(pop[i].acc - pop[i].test_acc)
		if dif < 0:
			dif = f'{dif:.4f}'
		else:
			dif = f'+{dif:.4f}'
		print(f'{pop[i].atts:3}   {pop[i].acc:.4f}   {pop[i].test_acc:.4f}   {dif}', end = '')
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

def set_pop_test_accs(pop, pop_size):
	for i in range(pop_size):
		pop[i].set_test_acc()
	return pop

def classifier(clf_option):
	if clf_option == 0:
		clf = GaussianNB()
	elif clf_option == 1:
		clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = None)
	elif clf_option == 2:
		clf = SVC(random_state = 0, max_iter = 50, gamma = 'auto')
	return clf

def split_data_target(file_name):
	dataset = pd.read_csv(file_name)
	data = dataset.iloc[:, :-1]
	target = dataset.iloc[:, -1]
	return data, target

def update_data():
	global train_data
	global test_data
	global column_names
	global chrom_size
	train_data = train_data.iloc[:, reduced_attributes]
	test_data = test_data.iloc[:, reduced_attributes]
	column_names = column_names[reduced_attributes]
	chrom_size = sum(reduced_attributes)

def calculate_int_pop_size():
	int_pop_size = int(pop_size*0.7)
	if int_pop_size%2 != 0:
		int_pop_size += 1
	return int_pop_size

def replace_dot(results):
	n_rows = len(results)
	n_elements = len(results[0])

	for r in range(n_rows):
		for e in range(n_elements):
			results[r][e] = str(results[r][e]).replace('.', ',')

	return results

def get_round_results(pop, pop_size, overfit = None):
	ind = pop[0]

	if overfit == False:
		for i in range(pop_size):
			dif = pop[i].acc - pop[i].test_acc
			if dif < 0.05:
				ind = pop[i]
				break

	ind_list = [
		Round,
		ind.atts,
		ind.acc,
		ind.test_acc,
		round_off(ind.acc - ind.test_acc),
		round_off(mean_acc(pop, pop_size)),
		round_off(Time),
		column_names[reduced_attributes]
	]

	if alg_option in fitness_options:
		ind_list.insert(4, round_off(ind.fitness))
	
	return ind_list

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
		last_acc = rounds_results[Round - 2][2]
		current_acc = rounds_results[Round - 1][2]
		last_acc = float(last_acc)
		current_acc = float(current_acc)

		if current_acc > last_acc:
			stagnant_round = 0
			return False
		elif current_acc < last_acc:
			return True
		else:
			last_atts = rounds_results[Round - 2][1]
			current_atts = rounds_results[Round - 1][1]
			if current_atts == last_atts:
				stagnant_round += 1
				if stagnant_round == max_stagnant_round:
					return True
			else:
				stagnant_round = 0
				return False

def save_results(results, columns, type_of_results = None):
	results = replace_dot(results)
	results_df = pd.DataFrame(results, columns = columns)

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

def get_alg_results():
	last_round = rounds_results_without_overfit[-1][0]

	best_round = last_round

	for i in range(last_round, -1, -1):
		if rounds_results_without_overfit[i - 1][-4] < 0.05:
			best_round = i
			break

	alg_results = [
		s,
		last_round,
		rounds_results_without_overfit[best_round - 1][0],
		rounds_results_without_overfit[best_round - 1][1],
		rounds_results_without_overfit[best_round - 1][2],
		rounds_results_without_overfit[best_round - 1][-5],
		rounds_results_without_overfit[best_round - 1][-4],
		total_time,
		rounds_results_without_overfit[best_round - 1][-1]
	]

	if alg_option in fitness_options:
		fitness = rounds_results_without_overfit[best_round - 1][3]
		alg_results.insert(5, fitness)

	return alg_results




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
pop_sizes = [20]
pop_size = int()

n_selected = 3

#mutation_rates = [0.02, 0.05, 0.10]
mutation_rates = [0.05]
mutation_rate = float()

max_stagnant_gen = 30
n_best_inds = 10
max_round = 15
att_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_final_inds = len(att_rates)

max_stagnant_round = 2
sampling = 15





rounds_results_columns = [
	'Round',
	'Nº of attributes',
	'Train accuracy',
	'Test accuracy',
	'Difference',
	'Mean accuracy',
	'Time',
	'Attributes'
]

alg_results_columns = [
	's',
	'Last round',
	'Best round',
	'Nº of attributes',
	'Train accuracy',
	'Test accuracy',
	'Difference',
	'Time',
	'Attributes'
]

if alg_option in fitness_options:
	rounds_results_columns.insert(4, 'Fitness')
	alg_results_columns.insert(5, 'Fitness')




for pop_size in pop_sizes:
	for mutation_rate in mutation_rates:

		alg_results = list()

		for s in range(1, sampling + 1):

			Times = list()
			rounds_results = list()
			rounds_results_without_overfit = list()

			train_data, train_target = split_data_target(train_file_name)
			test_data, test_target = split_data_target(test_file_name)

			column_names = train_data.columns
			chrom_size = len(column_names)
			int_pop_size = calculate_int_pop_size()

			stagnant_round = 0

			print(f'\ns: {s}')
			print(f'Population size: {pop_size}')
			print(f'Mutation rate: {mutation_rate}')
			print(f'Dataset: {dataset}\n')

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


				best_individuals = set_pop_test_accs(best_individuals, n_best_inds)
				best_individuals = sort_population(best_individuals, n_best_inds)
				Print(best_individuals, n_best_inds, type_of_pop = 'best')

				if alg_option in final_options:
					final_inds = generate_final_inds()
					final_inds = set_pop_test_accs(final_inds, n_final_inds)
					final_inds = sort_population(final_inds, n_final_inds)

					reduced_attributes = final_inds[0].bool_index

					end = time()
					Time = (end - start) / 60

					Print(final_inds, n_final_inds, type_of_pop = 'final')
					rounds_results.append(get_round_results(final_inds, n_final_inds, overfit = True))
					rounds_results_without_overfit.append(get_round_results(final_inds, n_final_inds, overfit = False))
				else:
					reduced_attributes = best_individuals[0].bool_index

					end = time()
					Time = (end - start) / 60

					rounds_results.append(get_round_results(best_individuals, n_best_inds, overfit = True))
					rounds_results_without_overfit.append(get_round_results(best_individuals, n_best_inds, overfit = False))

				print('rounds_results')
				print(rounds_results)
				print('rounds_results_without_overfit')
				print(rounds_results_without_overfit)
					
				print(f'Time(minute): {round_off(Time):.4f}\n')
				Times.append(Time)

				update_data()

				if verify_round_stagnation():
					break

			total_time = round_off(sum(Times))
			print(f'Total time(minute): {total_time:.4f}\n')

			alg_results.append(get_alg_results())


		save_results(alg_results, alg_results_columns, type_of_results = 'alg')
