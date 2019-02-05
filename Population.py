
MUTATION_PRINT = 0

# keeping a list of the innovations that occurred in the current generation
# it is possible to ensure that when the same structure arises more than once through independent
# mutations in the same generation, each identical mutation is assigned the
# same innovation number

# divide the population into species with similar topologies
# we need to match the topologies
# number of excess and disjoint genes between a pair of genomes is the measure of their compatibility
from Genome import Genome
from Genome import Node_type
from random import uniform
from random import randint
from NeuralNetwork import NeuralNetwork
from math import pow

import copy

from Genome import Connection_gene

class Population:
	
	def __init__(self, members, inputs, outputs):
		# dict of different nieches, where nieche ID -> nieche
		self.nieches = {}
		# Holding the genom of the population members, where genome ID -> genome
		self.genomes = {}
		# How many members the population has
		self.members = members

		self.inputs = inputs
		self.outputs = outputs

		# Mutation parameters:
		self.weight_mutation 			= 1
		# Mutation coeficients
		self.mutation_addConnection 	= 0.20
		self.mutation_addNode 			= 0.20
		self.mutation_changeConnection 	= 1.0
		self.mutation_EnDisConncetion 	= 0.20
		self.mutation_removeNode 		= 0.10
		self.mutation_addBias			= 0.20
		# Difference coeficients
		self.excess_constant			= 1
		self.disjoint_constant			= 1
		self.weight_diff_constant		= 0.1
		self.diff_threshold				= 1

		# How many of the nieche population dies in %
		self.how_many_dies = 75

		# Counts how many different innovations are, after a mutation
		self.innov_id = 0			# Global innovation number, basically the connection ID
		self.genome_id = 0			# Global Genome ID
		self.niecheID = 0			# Global Nieche ID
		self.diff_innov = dict()	# Storing the innovations occured so far

		self.create_members()

	############################################# PUBLIC

	# I have to clear the fitness because it is stored in a list, 
	# and the list is always just appending, causeing problems.
	def clear_fitness(self):
		for gkey in self.genomes:
			genome = self.genomes[gkey]
			genome.clear_fitness()

	# Evaluates the fitness for the whole population (Fi)
	# I could give the data as: Input1:[ [Input1t1], ... [Input1tN] ] Input2:[ [Input2t1], ... [Input2tN] ] ... InputN:[ [InputNt1], ... [InputNtN] ]
	# This could be an over complification, but im not sure, it could improve the possibilities of evaluating.
	# I might do that later, i think it would required to rewrite a lot of things.
	def evaluate_fitness(self, inputs, targets):
		# Evaluate fittnes for each genome
		for gkey in self.genomes:
			genome = self.genomes[gkey]

			# Create neural network
			nn = NeuralNetwork(genome)
			
			# Give it input data
			nn.add_input(inputs)

			# Check output data
			outputs = nn.get_output()

			# Evaluate fitness from output data
			error = 0
			for index in range(len(outputs)):
				error = error + abs( outputs[index] - targets[index] ) + 0.005 * ( len(genome.connection_genes) + len(genome.node_genes) )
				#error = error + ( (outputs[index] - targets[index]) ** 2 )

			genome.set_fitness(round(error, 1))

	# Select the best of the best
	def selection(self):
		#print("\tSelect the fittest____________________________________________________")
		# Carry over the nieches which are still competitive
		survivor_nieches = {}
		# The survivors of the artificial selection
		survivor_genomes = {}

		# Go through nieches, remove im-competent nieches, and genomes
		for nkey in self.nieches:
			nieche = self.nieches[nkey]
			
			# Get nieches which can be carried over
			if(nieche.get_competitive() == True):
				survivor_nieches[str(nieche.get_nieche_id())] = nieche
			else:
				# This nieche is not competetive, skip all its members
				continue

			# Dont remove genomes from nieches where there are 2 or less members.
			# When they are not allowed to create offsprings for 3 cycles, 
			# They will be removed anyway
			if(len(nieche.members_by_genome_ID) <= 2):
				# Add the genomes inside the nieche to the new genomes list
				for genomeID in nieche.members_by_genome_ID:
					survivor_genomes[str(genomeID)] = (self.genomes[str(genomeID)])
				continue

			# Get the members fitnesses and sort them, weak first order
			sorted_by_fittness = []
			for genome_ID in nieche.members_by_genome_ID:
				# get the fitness of this genome
				sorted_by_fittness.append( [genome_ID, self.genomes[str(genome_ID)].get_fitness()] )
			# Sort in reverse so we just select the first all-dead members
			sorted_by_fittness.sort(reverse=True, key=lambda x: x[1])

			# Get how many members should die
			death_count = round(nieche.get_member_numb()*(self.how_many_dies/100))
			# Ensures that there are always 2 members left
			if( (nieche.get_member_numb()-death_count) < 2 and death_count != 0):
				death_count += (death_count-nieche.get_member_numb())
			
			# The new members in the nieche
			new_nieche_member_IDs = []
			# Add the survivors to the nieche
			for index in range( len(sorted_by_fittness)-death_count ):
				genome_ID = sorted_by_fittness[index][0]
				survivor_genomes[str(genome_ID)] = (self.genomes[str(genome_ID)])
				new_nieche_member_IDs.append(genome_ID)
			
			# Removed the lowest members from the nieche, so update the nieche object
			nieche.members_by_genome_ID = new_nieche_member_IDs
		
		# Only store the survivors in the genomes list
		self.genomes = survivor_genomes
		# Carry over the survivors
		self.nieches = survivor_nieches

		# for nkey in self.nieches:
		# 	nieche = self.nieches[nkey]
			#print("\t\tNieche", nkey, nieche.members_by_genome_ID)
		
		# for gkey in self.genomes:
		# 	genome = self.genomes[gkey]
			#print("\t\tGenome", gkey, genome.niecheID)

	# Select the survivors and breed them
	# Must calculate the free slots for a nieche, by summing the nieche fitnesses, 
	# and divide the nieche fitness with that summ, 
	# then multiply this number with the overall free slots.
	
	# TODO Probably because of the Roundation it creates more genomes than it should have, try to fix that somehow.
	def populate(self):
		#print("\tCreate new members____________________________________________________")

		# Calculate the overall fitness of the nieches
		summfittnes = 0
		for nkey in self.nieches:
			nieche = self.nieches[nkey]
			summfittnes += nieche.nieche_fitness

		# Set how many free slot a nieche gets
		for nkey in self.nieches:
			nieche = self.nieches[nkey]
			self.calculate_free_slots(nieche, summfittnes)
		
		# Create new genomes
		for nkey in self.nieches:
			nieche = self.nieches[nkey]
			nieche_total_members = nieche.get_free_slots() + nieche.get_member_numb()

			parents = []
			parents = self.get_highest_fitness_combination(nieche, nieche.get_free_slots())

			#print("\t\tparents", parents)

			# If this nieche has only one element, we can create a new element by mutating the parent.
			if(len(parents) == 0):
				#print("\t\tNo parents create child by cloneing:", nieche.get_free_slots())
				# How many child do we have to create?
				for i in range(nieche.get_free_slots()):
					child = self.get_clone( self.genomes[str(nieche.members_by_genome_ID[0])] )
					# Add to nieche, set nieche ID, add to genomes
					self.add_to_nieche(child, nieche)
			
			# If the nieche has free slots after the crossover, 
			# means they were too few parents to fill all the slots, 
			# we can  do more crossover, as in the nature, more resources bigger family.
			while(nieche_total_members > nieche.get_member_numb()):
				#print("\t\tCreate more children:", (nieche_total_members-nieche.get_member_numb()))
				# Get parents, create a child, add child to system
				for parent in parents:
					parent1 = self.genomes[str(parent[0])]
					parent2 = self.genomes[str(parent[1])]
					child = self.get_child(parent1, parent2)
					# Add to nieche, set nieche ID, add to genomes
					self.add_to_nieche(child, nieche)

					#The limit was reached while parents are still could create more
					if(nieche_total_members <= nieche.get_member_numb()):
						#print("\t\tEnough!:", nieche_total_members-nieche.get_member_numb())
						break

		#print("\t\tGenomes:", len(self.genomes))

	# Decide which mutation are we going to execute on a genome
	def mutate(self):
		print("\tMutate members") if(MUTATION_PRINT) else 0

		# The member with the highest fitness should be left unchanged
		# Get the alpha genome
		best_genome = self.get_best_genome()

		mutation_summ = self.mutation_addConnection+self.mutation_addNode+self.mutation_changeConnection+self.mutation_EnDisConncetion+self.mutation_removeNode

		for gkey in self.genomes:
			genome = self.genomes[gkey]
			print("Genome:", gkey) if(MUTATION_PRINT) else 0

			# If it is the genome with the highest fitness value, dont change it.
			if(genome == best_genome):
				print("\tThis is the best genome, skip") if(MUTATION_PRINT) else 0
				continue
			
			rnd = uniform(0,mutation_summ)

			if(rnd >= 0 and rnd < self.mutation_addConnection):
				print("\tadd connection") if(MUTATION_PRINT) else 0
				genome.add_connection_mutation()
			elif(rnd >= self.mutation_addConnection and rnd < (self.mutation_addNode+self.mutation_addConnection)):
				print("\tadd node") if(MUTATION_PRINT) else 0
				genome.add_node_mutation()
			elif(rnd >= (self.mutation_addNode+self.mutation_addConnection) and rnd < (self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection)):
				print("\tchange weight") if(MUTATION_PRINT) else 0
				genome.mutate_connection_gene()
			elif(rnd >= (self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection) and rnd < (self.mutation_EnDisConncetion+self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection)):
				print("\ttoggle connection") if(MUTATION_PRINT) else 0
				genome.enable_disable_mutation()
			elif(rnd >= (self.mutation_EnDisConncetion+self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection) and rnd < (self.mutation_removeNode+self.mutation_EnDisConncetion+self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection)):
				print("\tremove node") if(MUTATION_PRINT) else 0
				genome.mutation_remove_node()
			elif(rnd >= (self.mutation_removeNode+self.mutation_EnDisConncetion+self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection) and rnd < (self.mutation_addBias+self.mutation_removeNode+self.mutation_EnDisConncetion+self.mutation_changeConnection+self.mutation_addNode+self.mutation_addConnection)):
				print("\tremove node") if(MUTATION_PRINT) else 0
				genome.add_bias_mutation()

		# Group the same innovations under the same innovation number
		self.group_innovations()

	# Puts genomes in the correct nieche
	def group_genes(self):
		#print("\tCreate nieches____________________________________________________")

		# Saves runtime by knowing which genomes were compared and not comparing them again.
		already_compared = {}
		# Compare genes to eachother
		for gkey in self.genomes:
			genome1 = self.genomes[gkey]
			# Similarity of genome to niches
			genome_similarity = []

			# Check how similar to every nieche
			for nkey in self.nieches:
				nieche = self.nieches[nkey]
				#print("\t\tGenome, Nieche:", gkey, nkey)

				# It was emptyied meanwhile the gkey cycle..
				if(nieche.empty() == True):
					#print("\t\tEmpty nieche!!!")
					continue

				diff = 0			# the difference summed up
				genome_numb = 0		# number of compared element

				# Check every element in the nieche
				for memberID in nieche.members_by_genome_ID:
					genome2 = self.genomes[str(memberID)]

					# If we already comapred this elements skipp the comparison and use the saved data
					key1 = str(genome1.get_genome_id())+" "+str(genome2.get_genome_id())
					key2 = str(genome2.get_genome_id())+" "+str(genome1.get_genome_id())

					if key1 in already_compared:
						# update the difference and genome number
						diff += already_compared[key1]
						genome_numb += 1
						continue

					if key2 in already_compared:
						# update the difference and genome number
						diff += already_compared[key2]
						genome_numb += 1
						continue

					# Get the similarity index, which is actually the difference index
					similarity = self.get_similarity(genome1, genome2)
					# Save this comparison to save time next time.
					already_compared[str(genome1.get_genome_id())+" "+str(genome2.get_genome_id())] = similarity
					
					# update the difference and genome number
					diff += similarity
					genome_numb += 1
				
				# Count the overall similarity of the genome to the nieches
				genome_similarity.append( [nieche.identifier, diff/genome_numb] )

				#print("\t\tSimilarity:", genome_similarity[-1])

			# Sort the similarity to nieches, and pick the first, 
			# which is the most similar (similarest) if that is above the threshold.
			genome_similarity.sort(key=lambda x: x[1])

			# Remove genome from its current nieche, it will get a new nieche anyway, 
			# it could be the same as it was, but it doesnt matter
			self.remove_member_from_nieche(genome1)

			# Create new nieche and put this genome in.
			if(genome_similarity[0][1] > self.diff_threshold):
				#print("\t\t__Create new nieche!")
				# Create new nieche
				self.nieches[str(self.niecheID)] = (Nieche(self.niecheID))
				# Add genome to this nieche
				self.nieches[str(self.niecheID)].members_by_genome_ID.append(genome1.get_genome_id())
				# Store which nieche this genome belongs
				genome1.set_nieche_id(self.niecheID)

				self.niecheID += 1
			# Put this genome in the most similar nieche
			else:
				#print("\t\t__Put into already existing nieche!")
				# Get nieche ID to add
				niecheID = genome_similarity[0][0]
				# Get nieche form ID
				nieche = self.nieches[str(niecheID)]
				# Add genome ID to nieche
				nieche.add_member(genome1.get_genome_id())
				# Store which nieche this genome belongs
				genome1.set_nieche_id(niecheID)

		#for nkey in self.nieches:
			#print("\t\tNieche:", nkey, self.nieches[nkey].members_by_genome_ID)

	# Computes the fitness of the nieches, 
	# Makes sense to do it after all the fitness values are added, 
	# assuming there are more fittness data
	# If there is only 1 fittness, it doesnt matter
	def evaluate_nieche_fitness(self):
		# Go through nieches
		for nkey in self.nieches:
			nieche = self.nieches[nkey]

			# Wait for the Selection to clear the nieches, 
			# i dont want to modify the data at different places
			if(nieche.empty() == True):
				continue

			nieche_fitness = 0	# The summarised fitness of the niece
			genome_number = 0	# How many genomes are in the nieche
			
			# Go through genomes and summ up their fitness
			for genome_ID in nieche.members_by_genome_ID:
				nieche_fitness += self.genomes[str(genome_ID)].get_fitness()
				genome_number += 1
				#print("\t\t", self.genomes[str(genome_ID)].get_fitness())
			
			# Set nieche fitness for nieches
			nieche.set_nieche_fitness(nieche_fitness/genome_number)

	############################################# PRIVATE

	def calculate_free_slots(self, nieche, sumfitness):
		free_slots = self.members - len(self.genomes)
		niecheFitness = nieche.nieche_fitness

		free_for_nieche = free_slots * (niecheFitness/sumfitness)
		nieche.set_free_slots(int(round(free_for_nieche, 0)))

	# When deleting nieche, we also must delete the genomes
	def delete_nieche(self, nieche_key):
		# Remove genomes which was part of this nieche.
		for genomeID in self.nieches[nieche_key].members_by_genome_ID:
			# delete this genome
			del self.genomes[str(genomeID)]
		
		# Remove the nieche
		del self.nieches[nieche_key]

	# Adds to nieche, sets genome's nieche ID, and adds to self.genomes
	# Addig a genome to nieche became compilcated so create a function, so i dont forget something.
	def add_to_nieche(self, new_genome, nieche):
		nieche.add_member(new_genome.get_genome_id())
		new_genome.set_nieche_id(nieche.identifier)
		self.genomes[str(new_genome.get_genome_id())] = (new_genome)

	# Removes the genome from the nieche
	def remove_member_from_nieche(self, genome):
		# Get which nieche this genome is
		niecheID = genome.get_nieche_id()
		# Search this nieche
		nieche = self.nieches[str(niecheID)]
		# Remove the member from the nieche
		nieche.remove_member(genome.get_genome_id())

	# Compares genomes to each other to get how similar they are, actually returns how different they are TODO change the name.
	def get_similarity(self, genome1, genome2):
		len1 = len(genome1.get_connection_genes())
		len2 = len(genome2.get_connection_genes())

		Norm = len1 if(len1>len2) else len2

		excess_gene = 0
		disjoint_gene = 0
		weight_diff = 0

		# Get the 1st genomes connections, 
		# and try to find something with the same connection ID in the secound genomes. 
		# If there is not a connection with this ID -> excess, 
		# If there is but expressed1 != expressed2 -> Disjoint, 
		# Else measure weight difference
		# Mark it down to whom i found a pair in genome2, 
		# because that is going to be also an excess gene

		conn_not_evaluated_in_g2 = len(genome2.get_connection_genes())
		for conn1 in genome1.get_connection_genes():
			excessGene = True
			disjointGene = False
			deltaWeight = 0
			for conn2 in genome2.get_connection_genes():
				# Both have connection with this innovation number
				if(conn1.get_innovation_number() == conn2.get_innovation_number()):
					excessGene = False
					conn_not_evaluated_in_g2 -= 1
					# The expressed parameter is not matching
					if(conn1.expressed != conn2.expressed):
						disjointGene = True
					# Get the deltaWeight, because everything seems to be in order
					else:
						deltaWeight= abs(conn1.get_weight()-conn2.get_weight())
			
			if(excessGene == True):
				excess_gene += 1
			elif(disjointGene == True):
				disjoint_gene += 1
			else:
				weight_diff += deltaWeight

		excess_gene += conn_not_evaluated_in_g2

		return ((self.excess_constant*excess_gene)/Norm) + ((self.disjoint_constant*disjoint_gene)/Norm) + (weight_diff * self.weight_diff_constant)

	# Get two genome and make new genome from them.
	def get_child(self, parent1, parent2):
		new_genome = Genome(weight_mutation=self.weight_mutation, input_nodes=self.inputs, output_nodes=self.outputs, genome_id=self.genome_id)
		self.genome_id += 1

		fitness1 = parent1.get_fitness()
		fitness2 = parent2.get_fitness()

		if(fitness1 > fitness2):
			genome1 = parent1
			genome2 = parent2
		else:
			genome1 = parent2
			genome2 = parent1

		for conn1 in genome1.get_connection_genes():
			copy_con1 = copy.deepcopy(conn1)

			excessGene = True
			disjointGene = False
			newConn = 0
			for conn2 in genome2.get_connection_genes():
				copy_con2 = copy.deepcopy(conn2)
				# Both have connection with this innovation number
				if(conn1.get_innovation_number() == conn2.get_innovation_number()):
					excessGene = False
					# The expressed parameter is not matching
					if(conn1.expressed != conn2.expressed):
						disjointGene = True
					# Get the deltaWeight, because everything seems to be in order
					else:
						newConn = copy_con1 if(randint(0,1) == 1) else copy_con2
			
			if(excessGene == True):
				new_genome.connection_genes.append(copy_con1)
			elif(disjointGene == True):
				new_genome.connection_genes.append(copy_con1)
			else:
				new_genome.connection_genes.append(newConn)

			#Add node
			#nodeIn, nodeOut = copy.deepcopy(new_genome.connection_genes[-1].get_connected_nodes())
			nodeInID, nodeOutID = copy.copy(new_genome.connection_genes[-1].get_connected_nodes_id())

			#Add node if there wasnt any node like this before
			if(new_genome.get_node_by_id(nodeInID) == False):
				node = copy.deepcopy(genome1.get_node_by_id(nodeInID))
				new_genome.node_genes.append(node)
			if(new_genome.get_node_by_id(nodeOutID) == False):
				node = copy.deepcopy(genome1.get_node_by_id(nodeOutID))
				new_genome.node_genes.append(node)

			# Get the higher global node id
			copy_gnID1 = copy.copy(parent1.global_node_id)
			copy_gnID2 = copy.copy(parent2.global_node_id)
			new_genome.global_node_id = copy_gnID1 if(copy_gnID1>copy_gnID2) else copy_gnID2

		return new_genome

	# Creates a child by cloning the parent, later they will be mutated
	def get_clone(self, genome):
		# Create a child
		child = Genome(weight_mutation=self.weight_mutation, input_nodes=self.inputs, output_nodes=self.outputs, genome_id=self.genome_id)
		self.genome_id += 1

		# Copy the parents genes to the child
		child.connection_genes = copy.deepcopy(genome.connection_genes)
		child.node_genes = copy.deepcopy(genome.node_genes)
		child.global_node_id = copy.copy(genome.global_node_id)

		# I dont have to mutate the child, it will be mutated all together.

		return child

	# Groups the same innovations under the same innovation ID over the Genomes
	def group_innovations(self):

		#Get the different innovations
		for gkey in self.genomes:
			genome = self.genomes[gkey]

			# This genome doesnt have a new innovation, no add node/connection mutation happened
			if(len(genome.get_new_innovation_ids()) == 0):
				continue

			# Get innovation id(s), possible to have 2 when add node mutation happened
			innovation_ids = genome.get_new_innovation_ids()
			
			# Check both innovations
			for innovation_id in innovation_ids:
				[node1_id, node2_id] = genome.get_connection_by_id(innovation_id).get_connected_nodes_id()
				
				# Unique innovations get a new contaion, non-unique will be put to their container
				key = str(node1_id) + " " + str(node2_id)

				# Already added
				if key in self.diff_innov:
					self.diff_innov[key].append([gkey, innovation_id, False])
				# New
				else:
					self.diff_innov[key] = []							# Going to be an array
					self.diff_innov[key].append(self.innov_id)			# The first element is the innovation ID that it should be
					self.diff_innov[key].append([gkey, innovation_id, False])	# The others are the genomes and their innovation numbers
					self.innov_id += 1

			# Clear new innovation IDs from Genome
			genome.clear_new_innovations()

		# Assign the same Innovation number to the same innovations, 
		for key in self.diff_innov:
			correct_innov = self.diff_innov[key][0]
			#Go through the elements in this array
			for element in self.diff_innov[key]:
				# If it is a list, the first element is not a list.
				if( isinstance(element, list) ):
					gkey = element[0]
					wrong_innov = element[1]
					already_done = element[2]

					if(already_done == True):
						continue

					# If this genome is still active
					if gkey in self.genomes:
						genome = self.genomes[gkey]
						genome.change_innovation_id(wrong_innov, correct_innov)
						# This will mark that we already changed this genome innovation ID, dont do it again.
						element[2] = True

	# Create the initial population
	def create_members(self):
		nkey = str(self.niecheID)
		self.nieches[nkey] = Nieche(self.niecheID)
		
		for i in range(self.members):
			# Create a new genome
			gkey = str(self.genome_id)
			self.genomes[gkey] = Genome(
				weight_mutation = self.weight_mutation,
				input_nodes = self.inputs,
				output_nodes = self.outputs,
				genome_id = self.genome_id)
			
			# Create input, output nodes, and connect them
			self.genomes[gkey].create_inputs()
			self.genomes[gkey].create_outputs()
			self.genomes[gkey].create_innitial_connections()
			
			# Tell which nieche this node is belongs to, at init all belongs to the same, the 0
			self.genomes[gkey].set_nieche_id(self.niecheID)

			# At init add every member to the first nieche/species
			self.nieches[nkey].add_member(self.genome_id)
			
			# Increment ID
			self.genome_id += 1

		#The newly created Genomes has init innovations, connection between outputs and inputs
		#which we need to group together
		self.group_innovations()

		self.niecheID += 1

	# Within a nieche returns the highest possible fitness values, to select parents
	def get_highest_fitness_combination(self, nieche, number):
		fitnesses = []
		already_calculated = {}

		for member1 in nieche.members_by_genome_ID:
			for member2 in nieche.members_by_genome_ID:
				
				# Check if we already calculated the fitness of these two,
				# If yes we can skip this.
				key1 = str(member1) + " " + str(member2)
				key2 = str(member2) + " " + str(member1)

				if( (key1 in already_calculated.keys()) or (key2 in already_calculated.keys()) ):
					continue

				# Two different genomes within the same nieche
				if(member1 != member2):
					already_calculated[key1] = 1

					fitness1 = self.genomes[str(member1)].get_fitness()
					fitness2 = self.genomes[str(member2)].get_fitness()

					fitnesses.append( [member1, member2, fitness1+fitness2] )

		fitnesses.sort(reverse=True, key=lambda x: x[2])

		return fitnesses[0:number]

	############################################# GETTER

	# Return the genomes of the population
	def get_genomes(self):
		return self.genomes

	# Returns the genome with the highest fitness value
	def get_best_genome(self):
		# The best genome, and its fitness value
		best = 0
		curr_highest_fitness = 0
		# Check every genome
		for gkey in self.genomes:
			genome = self.genomes[gkey]

			# If this has higher fitness value, update the variables
			if(genome.get_fitness() > curr_highest_fitness):
				curr_highest_fitness = genome.get_fitness()
				best = genome
		
		# Return the best genome
		return best

	############################################# SETTER

class Nieche:
	def __init__(self, ID):
		self.identifier = ID
		self.members_by_genome_ID = []
		self.nieche_fitness = 0
		# List to see how many offsprings were generated by this nieche, 
		# if 0 0 0 we can delete this nieche
		self.prev_free_slots = [1,1,1]
		self.competitive = True

	# Tells if nieche empty or not
	def empty(self):
		if( len(self.members_by_genome_ID) == 0 ):
			return True
		else:
			return False

	# Returns the number of genomes in the nieche
	def get_member_numb(self):
		return len(self.members_by_genome_ID)

	def set_nieche_fitness(self, fitness):
		self.nieche_fitness = fitness

	# Add how many offsprings this nieche's members can produce
	def set_free_slots(self,slots):
		# Remove the last element, and add the new element
		self.prev_free_slots.pop()
		self.prev_free_slots.insert(0, slots)

		# If the 
		if(self.prev_free_slots == [0,0,0]):
			# Nieche can be deleted
			self.competitive = False
	
	def get_free_slots(self):
		return self.prev_free_slots[0]

	def get_competitive(self):
		return self.competitive

	def get_nieche_id(self):
		return self.identifier

	# Adds a member to the 
	def add_member(self, genomeID):
		self.members_by_genome_ID.append(genomeID)

	# Removes a member from the nieche, faster
	def remove_member(self, genomeID):
		self.members_by_genome_ID.remove(genomeID)

	# Removes a member from the nieche
	def remove_member_slow(self, genomeID):
		index = self.get_index_by_genomeID(genomeID)
		self.members_by_genome_ID.remove(index)
	
	# Gets the members index in the self.members_by_genome_ID list
	def get_index_by_genomeID(self, genomeID):
		index = 0
		for member in self.members_by_genome_ID:
			if(member == genomeID):
				return index
			index +=1
