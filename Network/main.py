# Start message
print("Initalizing Neural-Network please wait...")

# Imports
import random
import os, sys
import numpy as np

# Clear function
CC = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

# create a neural network
class neural_network:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# weights between input and hidden layer
		self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes)
		# weights between hidden and output layer
		self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)

		# bias between input and hidden layer
		self.bias_ih = np.random.rand(self.hidden_nodes, 1)
		# bias between hidden and output layer
		self.bias_ho = np.random.rand(self.output_nodes, 1)

		# learning rate
		self.learning_rate = 0.1

	# sigmoid function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# derivative of sigmoid function
	def sigmoid_derivative(self, x):
		return x * (1 - x)

	# feed forward
	def feed_forward(self, input_array):
		# convert input array to 2d array
		inputs = np.array(input_array, ndmin=2).T

		# calculate the hidden layer
		hidden = np.dot(self.weights_ih, inputs)
		hidden = np.add(hidden, self.bias_ih)
		# apply activation function
		hidden = self.sigmoid(hidden)

		# calculate the output layer
		output = np.dot(self.weights_ho, hidden)
		output = np.add(output, self.bias_ho)
		# apply activation function
		output = self.sigmoid(output)

		return output

	# train the neural network
	def train(self, input_array, target_array):
		# convert input array to 2d array
		inputs = np.array(input_array, ndmin=2).T

		# convert target array to 2d array
		targets = np.array(target_array, ndmin=2).T

		# calculate the hidden layer
		hidden = np.dot(self.weights_ih, inputs)
		hidden = np.add(hidden, self.bias_ih)
		# apply activation function
		hidden = self.sigmoid(hidden)

		# calculate the output layer
		outputs = np.dot(self.weights_ho, hidden)
		outputs = np.add(outputs, self.bias_ho)
		# apply activation function
		outputs = self.sigmoid(outputs)

		# calculate the error
		output_errors = np.subtract(targets, outputs)

		# calculate the hidden layer error
		hidden_errors = np.dot(self.weights_ho.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.weights_ho += self.learning_rate * np.dot((output_errors * self.sigmoid_derivative(outputs)),
		                                               np.transpose(hidden))

		# update the bias for the links between the hidden and output layers
		self.bias_ho += self.learning_rate * output_errors * self.sigmoid_derivative(outputs)

		# update the weights for the links between the input and hidden layers
		self.weights_ih += self.learning_rate * np.dot((hidden_errors * self.sigmoid_derivative(hidden)),
		                                               np.transpose(inputs))

		# update the bias for the links between the input and hidden layers
		self.bias_ih += self.learning_rate * hidden_errors * self.sigmoid_derivative(hidden)

	# copy the neural network
	def copy(self):
		copy = neural_network(self.input_nodes, self.hidden_nodes, self.output_nodes)
		copy.weights_ih = self.weights_ih.copy()
		copy.weights_ho = self.weights_ho.copy()
		copy.bias_ih = self.bias_ih.copy()
		copy.bias_ho = self.bias_ho.copy()
		copy.learning_rate = self.learning_rate
		return copy

	# mutate the neural network
	def mutate(self, mutation_rate):
		# mutate the weights between the input and hidden layers
		for i in range(len(self.weights_ih)):
			for j in range(len(self.weights_ih[i])):
				if random.random() < mutation_rate:
					self.weights_ih[i][j] += random.uniform(-1, 1)

		# mutate the weights between the hidden and output layers
		for i in range(len(self.weights_ho)):
			for j in range(len(self.weights_ho[i])):
				if random.random() < mutation_rate:
					self.weights_ho[i][j] += random.uniform(-1, 1)

		# mutate the bias between the input and hidden layers
		for i in range(len(self.bias_ih)):
			if random.random() < mutation_rate:
				self.bias_ih[i] += random.uniform(-1, 1)

		# mutate the bias between the hidden and output layers
		for i in range(len(self.bias_ho)):
			if random.random() < mutation_rate:
				self.bias_ho[i] += random.uniform(-1, 1)


# create a population of neural networks
class population:
	def __init__(self, size, input_nodes, hidden_nodes, output_nodes):
		self.size = size
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes
		self.neural_networks = []
		for i in range(int(size)):
			self.neural_networks.append((input_nodes, hidden_nodes, output_nodes))

	# get the best neural network
	def get_best(self):
		best = self.neural_networks[0]
		for i in range(1, len(self.neural_networks)):
			if self.neural_networks[i].fitness > best.fitness:
				best = self.neural_networks[i]
		return best

	# get the average fitness
	def get_average_fitness(self):
		total = 0
		for i in range(len(self.neural_networks)):
			total += self.neural_networks[i].fitness
		return total / len(self.neural_networks)

	# get the total fitness
	def get_total_fitness(self):
		total = 0
		for i in range(len(self.neural_networks)):
			total += self.neural_networks[i].fitness
		return total

	# natural selection
	def natural_selection(self):
		new_population = []
		for i in range(len(self.neural_networks)):
			parent = self.select_parent()
			new_population.append(parent)
		self.neural_networks = new_population

	# select a parent
	def select_parent(self):
		index = 0
		r = random.uniform(0, self.get_total_fitness())
		while r > 0:
			r -= self.neural_networks[index].fitness
			index += 1
		index -= 1
		return self.neural_networks[index].copy()

	# crossover
	def crossover(self, parent1, parent2):
		child = neural_network(self.input_nodes, self.hidden_nodes, self.output_nodes)
		midpoint = random.randint(0, len(parent1.weights_ih))
		for i in range(len(parent1.weights_ih)):
			for j in range(len(parent1.weights_ih[i])):
				if i < midpoint:
					child.weights_ih[i][j] = parent1.weights_ih[i][j]
				else:
					child.weights_ih[i][j] = parent2.weights_ih[i][j]
		midpoint = random.randint(0, len(parent1.weights_ho))
		for i in range(len(parent1.weights_ho)):
			for j in range(len(parent1.weights_ho[i])):
				if i < midpoint:
					child.weights_ho[i][j] = parent1.weights_ho[i][j]
				else:
					child.weights_ho[i][j] = parent2.weights_ho[i][j]
		midpoint = random.randint(0, len(parent1.bias_ih))
		for i in range(len(parent1.bias_ih)):
			if i < midpoint:
				child.bias_ih[i] = parent1.bias_ih[i]
			else:
				child.bias_ih[i] = parent2.bias_ih[i]
		midpoint = random.randint(0, len(parent1.bias_ho))
		for i in range(len(parent1.bias_ho)):
			if i < midpoint:
				child.bias_ho[i] = parent1.bias_ho[i]
			else:
				child.bias_ho[i] = parent2.bias_ho[i]
		return child

	# mutate the population
	def mutate(self, mutation_rate):
		for i in range(len(self.neural_networks)):
			self.neural_networks[i].mutate(mutation_rate)

	# create the next generation
	def next_generation(self):
		self.natural_selection()
		new_population = []
		for i in range(len(self.neural_networks)):
			parent1 = self.select_parent()
			parent2 = self.select_parent()
			child = self.crossover(parent1, parent2)
			new_population.append(child)
		self.neural_networks = new_population
		self.mutate(0.1)


# Check the fitness of the neural network
def check_fitness(neural_network):
	fitness = 0
	for i in range(len(neural_network.weights_ih)):
		for j in range(len(neural_network.weights_ih[i])):
			if neural_network.weights_ih[i][j] > 0:
				fitness += 1
	for i in range(len(neural_network.weights_ho)):
		for j in range(len(neural_network.weights_ho[i])):
			if neural_network.weights_ho[i][j] > 0:
				fitness += 1
	for i in range(len(neural_network.bias_ih)):
		if neural_network.bias_ih[i] > 0:
			fitness += 1
	for i in range(len(neural_network.bias_ho)):
		if neural_network.bias_ho[i] > 0:
			fitness += 1
	if fitness < 100:
		network_grade = 'F'
	if fitness > 100:
		network_grade = 'D'
	if fitness > 300:
		network_grade = 'C'
	if fitness > 500:
		network_grade = 'B'
	if fitness > 1000:
		network_grade = 'A'
	if fitness > 2000:
		network_grade = 'A+'
	return 'Network Fitness: ' + str(fitness) + '\n' + "Network grade: " + network_grade

def check_weights(neural_network):
	for i in range(len(neural_network.weights_ih)):
		for j in range(len(neural_network.weights_ih[i])):
			if neural_network.weights_ih[i][j] > 0:
				return 'Weight:' + str(neural_network.weights_ih[i][j])


def check_bias(neural_network):
	for i in range(len(neural_network.bias_ih)):
		if neural_network.bias_ih[i] > 0:
			return 'Bias:' + str(neural_network.bias_ih[i])


# create a neural network (DEMO DON'T COPY IN CODE)
CC()
Selected_input_nodes = input('Input Nodes: ')
Selected_hidden_nodes = input('Hidden Nodes: ')
Selected_output_nodes = input('Output Nodes: ')
popYorN = input('Do you want to use a population of networks? Y or N: ')

if popYorN == 'y':
	pop_size = input('Population size: ')
	population = population(pop_size, Selected_input_nodes, Selected_hidden_nodes, Selected_output_nodes)
	print("Population Created")
elif popYorN == 'n':
	print("No population made")
else:
	print("Please enter Y or N")

neural_network = neural_network(int(Selected_input_nodes), int(Selected_hidden_nodes), int(Selected_output_nodes))
print("Master Network made... ")
print(check_fitness(neural_network))
print(check_weights(neural_network))
print(check_bias(neural_network))
print("Input:"+Selected_input_nodes+'\n'+"Hidden:"+Selected_hidden_nodes+'\n'+"Output:"+Selected_output_nodes)
