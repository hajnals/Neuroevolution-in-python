from enum import Enum
from random import randint
from random import uniform
from numpy.random import normal

class Genome:
	# Use member variable in population because when we will have 2 populations this will be incremented as heck
	static_innovation = 1
	
	def __init__(self, weight_mutation, input_nodes, output_nodes, genome_id):
		self.weight_mutation = weight_mutation
		self.input_nodes = input_nodes
		self.output_nodes = output_nodes
		self.genome_id = genome_id

		self.node_genes = []
		self.connection_genes = []
		self.global_node_id = 0
		
		# Shows which nieche the genome is, set by default to none
		self.niecheID = None

		# The fitness of the genome
		# The fitness is going to be a list of fitness values, 
		# and when the fitness of a genome is evaluated, the mean of the fitness will be returned
		self.fitness = []

		# This will keep track of the new innovation, must be reseted after grouping innovations, 
		# only one innovation could happen in cycle so the new innovation is always the last one
		# But there are cases when there was no new innovation because had no or different mutation
		self.new_innovation_ids = []

	# Returns the nieche ID
	def get_nieche_id(self):
		return self.niecheID
	
	# Sets which nieche this genome is belongs to, this fasten the removal of the genome from a nieche
	def set_nieche_id(self, niecheID):
		self.niecheID = niecheID

	# Returns the input connections and weight for a given node
	def get_connections_for_node(self, nodeID):
		node_connections = []

		# Collect input nodes and connections for a given node
		for connection in self.connection_genes:
			[nodeIn,nodeOut] = connection.get_connected_nodes_id()
			if(nodeOut == nodeID):
				node_connections.append(connection)

		return node_connections

	# List the possible connections within this genome
	def possible_node_connections(self):
		possible_conn_list = []
		for node1 in self.node_genes:

			# we dont connect output nodes to anywhere
			if(node1.get_node_type() == Node_type.OUTPUT):
				continue

			for node2 in self.node_genes:
				# dont connect with itself
				if(node1.node_id == node2.node_id):
					continue
				
				# dont connect if both types are the same and they are input or ouput nodes
				# you can connect two hidden nodes
				if	(	(node1.get_node_type() == node2.get_node_type()) 
					and (node1.get_node_type() == Node_type.INPUT or node1.get_node_type() == Node_type.OUTPUT)
					):
					continue

				# If a connection already exist, either way, they are directional a0-->a1
				if(self.check_if_connection_exist(node1.get_node_id(), node2.get_node_id())):
					continue
				
				# Dont connect an hidden node to an input node
				if(node1.get_node_type() == Node_type.HIDDEN and node2.get_node_type() == Node_type.INPUT):
					continue

				# If every test was successfull that means these nodes can be connected!
				possible_conn_list.append([node1, node2])

		return possible_conn_list

	# Change the innovation ID of a connection
	def change_innovation_id(self, old_id, new_id):
		connection = self.get_connection_by_id(old_id)
		connection.change_innovation_number(new_id)

	# Finds a connection by ID
	def get_connection_by_id(self, innovation_id):
		for connection in self.connection_genes:
			if(connection.get_innovation_number() == innovation_id):
				return connection

		# Cannot find this ID
		return False

	# Finds a node by ID
	def get_node_by_id(self, node_id):
		for node in self.node_genes:
			if(node.get_node_id() == node_id):
				return node

		return False

	# Creates input nodes
	def create_inputs(self):
		for i in range(self.input_nodes):
			self.node_genes.append( Node_gene(
				node_type = Node_type.INPUT,
				node_id = self.global_node_id ))
			
			self.global_node_id += 1

	# Creates output nodes
	def create_outputs(self):
		for i in range(self.output_nodes):
			self.node_genes.append( Node_gene(
				node_type = Node_type.OUTPUT,
				node_id = self.global_node_id ))
			
			self.global_node_id += 1

	# Connect input and output nodes at initialization
	def create_innitial_connections(self):
		for input_node in self.node_genes:
			if(input_node.get_node_type() == Node_type.INPUT):
				for output_node in self.node_genes:
					if(output_node.get_node_type() == Node_type.OUTPUT):
						self.connection_genes.append(Connection_gene(
							in_node=input_node.node_id,
							out_node=output_node.node_id,
							weight=uniform(-1,1),
							expressed=True,
							innovation_number=Genome.static_innovation))
						
						# Add this innovation to the newly created innovation list
						self.new_innovation_ids.append(Genome.static_innovation)
						Genome.static_innovation += 1

	# Get connections
	def get_connection_genes(self):
		return self.connection_genes

	# Get nodes
	def get_node_genes(self):
		return self.node_genes

	# Get the ID of the genome, not sure if necessary
	def get_genome_id(self):
		return self.genome_id

	def add_bias_mutation(self):
		#Find a hidden node to add the bias to
		possibleNodes = []
		for node in self.node_genes:
			# TODO should also check if already has a bias or not.
			if(node.get_node_type() == Node_type.HIDDEN):
				possibleNodes.append(node)

		rndNodeIndex = randint(0, len(possibleNodes)-1)
		rndNode = possibleNodes[rndNodeIndex]
		
		#Create the bias node
		biasNode = Node_gene(Node_type.HIDDEN, self.global_node_id)
		self.node_genes.append(biasNode)
		self.global_node_id += 1

		#Connect the bias node with the node
		biasConn = Connection_gene(
				in_node = biasNode,
				out_node = rndNode,
				weight = normal(0, self.weight_mutation),			# Otherwise it would have too much impact and die early
				expressed = True,
				innovation_number = Genome.static_innovation )
		
		self.connection_genes.append(biasConn)
		self.new_innovation_ids.append(Genome.static_innovation)
		Genome.static_innovation += 1

	# Adds a connection to the topology
	def add_connection_mutation(self):
		# Add a new non existing connection between nodes
		# Should be checked if the 2 nodes have normal or backwards connection. we dont want a connection to both ways
		# Eg: connect 2-4 check if 2-4 is exist or 4-2 exist.

		# Get possible node connections
		possible_connections = self.possible_node_connections()

		if( len(possible_connections) == 0 ):
			return

		rnd_conn = randint(0, (len(possible_connections)-1) )
		[node1, node2] = possible_connections[rnd_conn]

		reverse_order = False
		if((node1.get_node_type() == Node_type.OUTPUT) and ((node2.get_node_type() == Node_type.HIDDEN) or (node2.get_node_type() == Node_type.INPUT))):
			reverse_order = True
		if(node1.get_node_type() == Node_type.HIDDEN and node2.get_node_type() == Node_type.INPUT ):
			reverse_order = True

		#connect the two node
		self.connection_genes.append(
			Connection_gene(
				in_node = node1.node_id if(reverse_order == False) else node2.node_id,
				out_node = node2.node_id if(reverse_order == False) else node1.node_id,
				weight = normal(0, self.weight_mutation),		# Otherwise it would have too much impact and die early
				expressed = True,
				innovation_number = Genome.static_innovation ))

		# Add this innovation to the newly created innovation list
		self.new_innovation_ids.append(Genome.static_innovation)
		Genome.static_innovation += 1

	# Change a connection's weight
	def mutate_connection_gene(self):
		possible_conn_list = []
		# Possible connections to modify
		for connection in self.connection_genes:
			# If it is disabled skip it.
			if(connection.expressed == False):
				continue
			possible_conn_list.append(connection)

		# If there is no possible connections to modify, return
		if(len(possible_conn_list) == 0):
			return

		#select a connection gene randomly, and change its value randomly.
		random_index = randint(0, (len(possible_conn_list)-1) )
		random_connection = possible_conn_list[random_index]

		current_weight = random_connection.get_weight()
		# Mutate with a random number between +weight_mutation and -weight_mutation
		mutated_weight = normal(current_weight, self.weight_mutation)

		random_connection.change_weight(mutated_weight)
	
	# Adds a node to the topology
	def add_node_mutation(self):

		# Get possible connections which can be divided by a node
		possible_conn_list = self.connection_genes

		# If there is no possible connections to modify, return
		if(len(possible_conn_list) == 0):
			return

		#get a connection gene to split into half
		random_index = randint(0, (len(possible_conn_list)-1) )
		old_connection = possible_conn_list[random_index]
		
		#disable this connection gene
		old_connection.disable_connection()
		
		#get connected nodes
		node_start_id, node_end_id = old_connection.get_connected_nodes_id()

		#get the old connections weight
		old_weight = old_connection.get_weight()
		
		#Add new node gene
		new_node = Node_gene(node_type=Node_type.HIDDEN, node_id=self.global_node_id)
		self.global_node_id += 1
		self.node_genes.append(new_node)
		
		#create a connection between the new node and the end node.
		from_new_node = Connection_gene(
			in_node = new_node.node_id,
			out_node = node_end_id,
			weight = old_weight,								# New connection leading out receives the old connection weight
			expressed = True,
			innovation_number = Genome.static_innovation )

		self.connection_genes.append(from_new_node)
		
		# Add this innovation to the newly created innovation list
		self.new_innovation_ids.append(Genome.static_innovation)
		Genome.static_innovation += 1

		#reate a connection between the new node and the start node.
		to_new_node = Connection_gene(
			in_node = node_start_id,
			out_node = new_node.node_id,
			weight = 1,											# New connection to new node receives a weight of 1
			expressed = True,
			innovation_number = Genome.static_innovation )

		self.connection_genes.append(to_new_node)
		
		# Add this innovation to the newly created innovation list
		self.new_innovation_ids.append(Genome.static_innovation)
		Genome.static_innovation += 1

	# Turn on/off weights
	def enable_disable_mutation(self):
		# Possible connections to enable/disable
		possible_conn_list = self.connection_genes

		# If there is no possible connections to modify, return
		if(len(possible_conn_list) == 0):
			return

		#select a random connection
		random_index = randint(0, (len(possible_conn_list)-1) )
		connection = possible_conn_list[random_index]

		if(connection.expressed == True):
			connection.disable_connection()		#disable this connection
			pass
		else:
			connection.enable_connection()		#enable this connection
			pass

	# Removes a node, can only be a hidden node, Might not use it as it seems a little harsh, what happens with the node without connections?
	def mutation_remove_node(self):
		# Get a node to remove
		possible_nodes = []
		for node in self.node_genes:
			if(node.get_node_type() == Node_type.HIDDEN):
				possible_nodes.append(node)

		# If we were not able to execute this mutation, notify the caller
		if(len(possible_nodes) == 0):
			return False

		random_index = randint(0, (len(possible_nodes)-1) )
		node_to_remove = possible_nodes[random_index]

		# Delete node from node list
		del(self.node_genes[self.node_genes.index(node_to_remove)])

		# Copy only the connections that we need
		new_list = []
		for connection in self.connection_genes:
			nodeIn_id, nodeOut_id = connection.get_connected_nodes_id()
			if( (nodeIn_id == node_to_remove.get_node_id()) or (nodeOut_id == node_to_remove.get_node_id()) ):
				pass
			else:
				new_list.append(connection)

		self.connection_genes = []
		self.connection_genes = new_list

		# We were able to remove a node and its connecitons
		return True

	# Check if two connected either way, Eg: 1--4 or 4--1
	def check_if_connection_exist(self, node1_id, node2_id):
		connection_exist = False

		for connection in self.connection_genes:
			cmp_node1_id,cmp_node2_id = connection.get_connected_nodes_id()

			# If the connection is existing in either way, normal or reversed
			if(		(node1_id == cmp_node1_id and node2_id == cmp_node2_id) 
				or 	(node1_id == cmp_node2_id and node2_id == cmp_node1_id)
				):
				connection_exist = True
				break

		return connection_exist

	# Get the new innovations
	def get_new_innovation_ids(self):
		return self.new_innovation_ids

	# Clear new innovation Array, Must be done after mutations
	def clear_new_innovations(self):
		self.new_innovation_ids = []

	# Return fitness value
	def get_fitness(self):
		fitness_numb = len(self.fitness)

		if(fitness_numb == 0):
			return 0
		
		summFitness = 0
		for fitness in self.fitness:
			summFitness += fitness

		return (1/(summFitness/fitness_numb))
	
	# Set fitness value
	def set_fitness(self,fitness):
		self.fitness.append(fitness)

	# Clears the fitness list
	def clear_fitness(self):
		self.fitness = []

	# Print Genes
	def print_genome(self):
		print("\n\t", self.genome_id)
		for connection in self.connection_genes:
			print("\t",connection.get_innovation_number(), connection.get_connected_nodes_id(), connection.get_weight(), connection.expressed)
		for node in self.node_genes:
			print("\t", node.get_node_id(), node.get_node_type())

############################################################################################
class Node_type(Enum):
	INPUT = 1
	HIDDEN = 2
	OUTPUT = 3

class Node_gene:
	def __init__(self, node_type, node_id):
		self.node_type = node_type
		self.node_id = node_id
		self.value = 0
		self.updated = False

	def get_node_type(self):
		return self.node_type

	def get_node_id(self):
		return self.node_id
	
	def get_updated(self):
		return self.updated
	
	def set_value(self, value):
		self.value = value
	
	def get_value(self):
		return self.value

	def set_updated(self, updated):
		self.updated = updated

############################################################################################
class Connection_gene:
	def __init__(self, in_node, out_node, weight, expressed, innovation_number):
		# There are nodes
		self.in_node = in_node
		self.out_node = out_node

		self.weight = weight
		self.expressed = expressed					# Enabled or not
		self.innovation_number = innovation_number

	def change_weight(self, weight):
		self.weight = weight

	def get_innovation_number(self):
		return self.innovation_number

	def change_innovation_number(self, new_innovation_number):
		self.innovation_number = new_innovation_number

	def get_in_node_id(self):
		return self.in_node

	def get_connected_nodes_id(self):
		return [self.in_node, self.out_node]

	def get_weight(self):
		return self.weight

	def disable_connection(self):
		self.expressed = False
	
	def enable_connection(self):
		self.expressed = True
	
	def print_conn(self):
		print("In node", self.in_node.node_id, self.in_node.node_type)
		print("Out node", self.out_node.node_id, self.out_node.node_type)
		print("Weight", self.weight)
		print("Expressed", self.expressed)
		print("Innovation", self.innovation_number)