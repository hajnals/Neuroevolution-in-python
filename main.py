from NeuralNetwork import NeuralNetwork
from Population import Population
from DrawNN import DrawNN
from random import shuffle
import matplotlib.pyplot as plt
import random

#Testing
from Genome import Genome
from Genome import Connection_gene
from Genome import Node_gene
from Genome import Node_type
from random import uniform

def createGraphs(p1, i, training_data):
	# The best nieche
	genome = p1.get_best_genome()
	print("Fitness:", genome.get_fitness())
	DrawNN(genome)
	# input-output function of the genome
	nn = NeuralNetwork(genome)
	x_axis = []
	neuron_outputs = []
	targets = []
	errors = []
	for data in training_data:
		nn.add_input(data[0])
		outputs = nn.get_output()
		errors.append(abs(outputs[0] - data[1][0]) )

		x_axis.append(data[0][0])
		neuron_outputs.append(outputs[0])
		targets.append(data[1][0])
	
	plt.plot(x_axis, neuron_outputs, 'b.', label='Output')
	plt.plot(x_axis, targets, 'r.', label='Target')
	plt.plot(x_axis, errors, 'g.', label='Error')
	plt.legend()
	plt.savefig('Fitness/map'+str(genome.get_genome_id())+'.png')
	plt.close()

	# Of the histogram of everyone
	# fitnessList = []
	# for gkey in p1.get_genomes():
	# 	fitness = p1.genomes[gkey].get_fitness()
	# 	fitnessList.append(fitness)

	# plt.hist(fitnessList, bins=50, color='g', density=True)
	# plt.savefig('Fitness/fitness'+ str(i) +'.png', bbox_inches='tight')
	# plt.close()

	# Pie chart of nieches
	# bestNieche = p1.get_best_genome().niecheID
	# labels = []
	# sizes = []
	# explode = []
	# for niecheKey in p1.nieches:
	# 	labels.append(p1.nieches[niecheKey].identifier)
	# 	sizes.append(p1.nieches[niecheKey].get_member_numb())
	# 	# Explode the best nieche
	# 	if(bestNieche == int(niecheKey)):
	# 		explode.append(0.1)
	# 	else:
	# 		explode.append(0)
	
	# figPie, axPie = plt.subplots()
	# axPie.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
	# axPie.axis('equal')
	# plt.savefig('Fitness/nieche' + str(i) +'.png')
	# plt.close()

	# Scatter plot
	x = []
	fitness = []
	index = 0
	color = []
	area = []
	for genomeKey in p1.genomes:
		x.append(index)
		fitness.append(p1.genomes[genomeKey].get_fitness())
		color.append(p1.genomes[genomeKey].niecheID)
		area.append(5*len(p1.genomes[genomeKey].connection_genes))
		index += 1
	
	plt.scatter(x,fitness,s=area,c=color, alpha=0.5)
	plt.savefig('Fitness/scatter' + str(i) +'.png')
	plt.close()

	pass

def main():
	# Set seed to always get the same result, for debugging
	#random.seed(49843651)

	# Create population of neural networks
	p1 = Population(members=100, inputs=1, outputs=1)

	# Create training data
	training_data = []
	for j in range(1,25):
		training_data.append([ [pow(j,2)], [j] ])
	
	for i in range(1,100):
		print("Evolution cycle:", i)
		
		# Test it fitness by asking them to first 100 integer root value
		print("\tEvaluate...")

		# Clear previous fitness values
		p1.clear_fitness()
		
		# Shuffle the data
		random.shuffle(training_data)

		#Feed the data
		for data in training_data:
			p1.evaluate_fitness(inputs=data[0], targets=data[1])

		print("\tMakeing figures...")
		createGraphs(p1, i, training_data)

		# Doing the rest
		p1.selection()
		p1.populate()
		p1.mutate()
		p1.group_genes()

		print(len(p1.genomes))
		print(len(p1.nieches))
	pass

# For testing functions
def test():

	pop1 = Population(members=100, inputs=1, outputs=1)

	# Create training data
	training_data = []
	for j in range(1,50):
		training_data.append([ [j**2], [j] ])	#Input array, Target array

	print(training_data)

	# Evolution cycle
	#for i in range(0,100):
	i = 0
	while(True):
		print("i", i)

		# Evaluate the fitness of the genomes
		pop1.clear_fitness()
		for data in training_data:
			pop1.evaluate_fitness(inputs=data[0], targets=data[1])

		# Evaluate the fitness of the nieches too
		pop1.evaluate_nieche_fitness()
		# Make figures..
		createGraphs(pop1, i, training_data)
		# Select the genomes we wish to keep
		pop1.selection()
		# Create new members
		pop1.populate()
		# Mutate the genomes
		pop1.mutate()
		# Put the newly created genes into species
		pop1.group_genes()

		i = i+1

	pass

def test2():
	genome = Genome(weight_mutation=0.1, input_nodes=2, output_nodes=1, genome_id = 1)
	# Create input, output nodes, and connect them
	genome.create_inputs()
	genome.create_outputs()

	#Create test data
	testData = []
	for x in range(0,100):
		testData.append([x,x])

	# Create 2 hidden layers, and fully connect them to test if it is still going to be non-linear
	h1 = Node_gene(node_type=Node_type.HIDDEN, node_id=10)
	h2 = Node_gene(node_type=Node_type.HIDDEN, node_id=11)
	h3 = Node_gene(node_type=Node_type.HIDDEN, node_id=12)
	h4 = Node_gene(node_type=Node_type.HIDDEN, node_id=13)

	i1 = genome.node_genes[0]
	i2 = genome.node_genes[1]

	o1 = genome.node_genes[2]

	b1 = Node_gene(node_type=Node_type.HIDDEN, node_id=14)
	b1.set_value(50)
	b2 = Node_gene(node_type=Node_type.HIDDEN, node_id=15)
	b2.set_value(50)
	b3 = Node_gene(node_type=Node_type.HIDDEN, node_id=16)
	b3.set_value(50)
	b4 = Node_gene(node_type=Node_type.HIDDEN, node_id=17)
	b4.set_value(50)

	genome.node_genes.append(h1)
	genome.node_genes.append(h2)
	genome.node_genes.append(h3)
	genome.node_genes.append(h4)

	genome.node_genes.append(b1)
	genome.node_genes.append(b2)
	genome.node_genes.append(b3)
	genome.node_genes.append(b4)

	#Connection i1 to h1
	con_i1_h1 = Connection_gene(
			in_node = i1,
			out_node = h1,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 1 )
	#Connection i2 to h1
	con_i2_h1 = Connection_gene(
			in_node = i2,
			out_node = h1,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 2 )
	#Connection i1 to h2
	con_i1_h2 = Connection_gene(
			in_node = i1,
			out_node = h2,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 3 )
	#Connection i2 to h2
	con_i2_h2 = Connection_gene(
			in_node = i2,
			out_node = h2,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 4 )
	#Connection h1 to h3
	con_h1_h3 = Connection_gene(
			in_node = h1,
			out_node = h3,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 5 )
	#Connection h2 to h3
	con_h2_h3 = Connection_gene(
			in_node = h2,
			out_node = h3,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 6 )
	#Connection h1 to h4
	con_h1_h4 = Connection_gene(
			in_node = h1,
			out_node = h4,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 7 )
	#Connection h2 to h4
	con_h2_h4 = Connection_gene(
			in_node = h2,
			out_node = h4,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 8 )
	#Connection h3 to o1
	con_h3_o1 = Connection_gene(
			in_node = h3,
			out_node = o1,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 9 )
	#Connection h4 to o1
	con_h4_o1 = Connection_gene(
			in_node = h4,
			out_node = o1,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 10 )
	#Conneciton b1-h1
	con_b1_h1 = Connection_gene(
			in_node = b1,
			out_node = h1,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 11 )
	#Conneciton b2-h2
	con_b2_h2 = Connection_gene(
			in_node = b2,
			out_node = h2,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 12 )
	#Conneciton b3-h3
	con_b3_h3 = Connection_gene(
			in_node = b3,
			out_node = h3,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 13 )
	#Conneciton b4-h4
	con_b4_h4 = Connection_gene(
			in_node = b4,
			out_node = h4,
			weight = uniform(-1, 1),
			expressed = True,
			innovation_number = 14 )

	genome.connection_genes.append(con_i1_h1)
	genome.connection_genes.append(con_i2_h1)
	genome.connection_genes.append(con_i1_h2)
	genome.connection_genes.append(con_i2_h2)
	genome.connection_genes.append(con_h1_h3)
	genome.connection_genes.append(con_h2_h3)
	genome.connection_genes.append(con_h1_h4)
	genome.connection_genes.append(con_h2_h4)
	genome.connection_genes.append(con_h3_o1)
	genome.connection_genes.append(con_h4_o1)

	genome.connection_genes.append(con_b1_h1)
	genome.connection_genes.append(con_b2_h2)
	genome.connection_genes.append(con_b3_h3)
	genome.connection_genes.append(con_b4_h4)
	
	
	# Draw and print
	DrawNN(genome)
	genome.print_genome()

	#Test
	o1_output = []
	i1_input = []
	i2_input = []
	h1_output = []
	h2_output = []
	h3_output = []
	h4_output = []
	for data in testData:
		nn = NeuralNetwork(genome)
		nn.add_input(data)
		nn.get_output()

		i1_input.append( genome.node_genes[0].value )
		i2_input.append( genome.node_genes[1].value )
		o1_output.append( genome.node_genes[2].value )
		h1_output.append( genome.node_genes[3].value )
		h2_output.append( genome.node_genes[4].value )
		h3_output.append( genome.node_genes[5].value )
		h4_output.append( genome.node_genes[6].value )
	
	# Plot
	#plt.plot(i1_input, i1_input, 'b-', label="i1_input")
	#plt.plot(i1_input, i2_input, 'g-', label="i2_input")
	#plt.plot(i1_input, h1_output, 'c.', label="h1_output")
	#plt.plot(i1_input, h2_output, 'm.', label="h2_output")
	plt.plot(i1_input, h3_output, 'y.', label="h3_output")
	plt.plot(i1_input, h4_output, 'k.', label="h4_output")
	plt.plot(i1_input, o1_output, 'r-', label="o1_output")
	plt.legend()

	plt.show()

# Entry point
if __name__ == "__main__":
	test()

