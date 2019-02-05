from Genome import Node_type
from Genome import Genome
import math
import numpy

PRINT_ENABLE = 0

# Instead of sigmoid function i should really use ReLu or leaking ReLu

def sigm(x):
	return ( 1 / (1 + math.exp(-x)) )

def leakyReLu(x, c1, c2):
	if(x>0):
		return x*c1
	else:
		return x*c2

class NeuralNetwork:
	def __init__(self, genome):
		self.genome = genome

	def add_input(self, input):
		i = 0
		for node in self.genome.get_node_genes():
			if(node.get_node_type() == Node_type.INPUT):
				# Add value to node, maybe it has no value member variable yet, so create it also.
				node.set_value(input[i])
				# Set the updated flag to true
				node.set_updated(True)
				i += 1
	
	def get_output(self):
		self.calcualte_node_values()

		# Return the value of the output nodes
		return_list = []
		for node in self.genome.get_node_genes():
			if(node.get_node_type() == Node_type.OUTPUT):
				return_list.append(node.get_value())
		
		# Set every node to not updated
		for node in self.genome.get_node_genes():
			node.set_updated(False)

		return return_list

	# Calculates the value of the nodes
	def calcualte_node_values(self):
		# Storing the values and weights for each node
		w_and_a_dict = {}
		# we have to evalue these nodes
		remaining_nodes = self.genome.get_node_genes()
		
		print("Gene:", self.genome.get_genome_id()) if(PRINT_ENABLE) else 0
		while( len(remaining_nodes) ):
			# Temporary array for stroing remaining nodes
			temp_list = []
			# Go through hidden and ouput nodes to see if i can calculate their values
			for node in remaining_nodes:
				print("\tnode:", node.get_node_id()) if(PRINT_ENABLE) else 0
				# Mark if we have to evaluate another node
				find_other_node = False

				# Output or Hidden node
				if(node.get_node_type() == Node_type.HIDDEN or node.get_node_type() == Node_type.OUTPUT):
					# Storing the weights and nodes values that are connected to this node
					w_and_a_dict[node.get_node_id()] = []
					# Get connections
					connections = self.genome.get_connections_for_node(node.get_node_id())

					# Go through this node's connections
					for connection in connections:
						print("\t\tInput connection:", connection.get_connected_nodes_id()) if(PRINT_ENABLE) else 0
						
						# If connection is not experessed skip this.
						if(connection.expressed == False):
							print("\t\t\tNot expressed") if(PRINT_ENABLE) else 0
							# Remove
							continue	# next connection

						# Get the weigh and the value of the connected node
						weight = 0
						value = 0
						inNode = self.genome.get_node_by_id( connection.get_in_node_id() )
						# If its value is updated
						if(inNode.get_updated() == True):
							# Get its weight and value
							value = inNode.get_value()
							weight = connection.get_weight()
							print("\t\t\tid, value, weight:", node.get_node_id(), [value, weight]) if(PRINT_ENABLE) else 0
							w_and_a_dict[node.get_node_id()].append([value, weight])
						# we cannot calculate this nodes value, come back later
						else:
							print("\t\t\tCan not calculate this connection!") if(PRINT_ENABLE) else 0
							find_other_node = True
							break
					
					# It has no connections, sometimes happen, keep old value, It is like a bias
					# Output is not alowed to keep old value
					# TODO this may be unecessary because it will run into the other condition where he does the same
					if(len(connections) == 0 and node.get_node_type() == Node_type.HIDDEN):
						print("\t\tBias node:") if(PRINT_ENABLE) else 0
						node.set_updated(True)
					
					# IF it has no connection and it is an output node set the value to 0
					# TODO this may be unecessary because it will run into the other condition where he does the same
					if(len(connections) == 0 and node.get_node_type() == Node_type.OUTPUT):
						print("\t\tOutput without connections") if(PRINT_ENABLE) else 0
						node.set_value(0)
						node.set_updated(True)

					# We cannot determine this nodes value, because we are missing inputs nodes
					if(find_other_node == True):
						# Mark this node, that we should come back later.
						temp_list.append(node)
						continue	# next node
					
					# We can calculate a value and assign to it.
					else:
						print("\t\tCount new value for node") if(PRINT_ENABLE) else 0
						node_value = 0

						# If weight and node list is empty, means every connection is not expressed
						if(len(w_and_a_dict[node.get_node_id()]) == 0 and node.get_node_type() == Node_type.HIDDEN):
							print("\t\tBias node:", node.get_value()) if(PRINT_ENABLE) else 0
							#Keep old value
							node.set_updated(True)
							continue	# next node

						# If weight and node list is empty, means every connection is not expressed, set the output value to 0
						if(len(w_and_a_dict[node.get_node_id()]) == 0 and node.get_node_type() == Node_type.OUTPUT):
							print("\t\tAssign 0:", node.get_value()) if(PRINT_ENABLE) else 0
							node.set_value(0)
							node.set_updated(True)
							continue
						
						# Multiply every connection with its weight, and summ them
						for value_and_weight in w_and_a_dict[node.get_node_id()]:
							value, weight = value_and_weight
							node_value += value*weight
						
						# If hidden node, use non-linearity, sigmoid, relu, tanh, etc..
						if(node.get_node_type() == Node_type.HIDDEN):
							# Get the non-linearity of this summary and pass as new value for the node
							node.set_value( numpy.tanh(node_value) )
							#node.set_value( sigm(node_value) )
							#node.set_value( leakyReLu(node_value, 1, 0.01) )
							print("\t\tNew hidden value:", node_value, "->", node.get_value()) if(PRINT_ENABLE) else 0
							node.set_updated(True)
						
						# If output node, depends on the problem, classification or regression.
						elif(node.get_node_type() == Node_type.OUTPUT):
							# Use normalization funciton according to problem.
							node.set_value(node_value)
							print("\t\tNew ouput value:", node.get_value()) if(PRINT_ENABLE) else 0
							node.set_updated(True)

			# If we have to evluate the same amount of node as before, this means we are stuck in an endless
			# loop caused by a deadlock, in this case we cannot determine the output value.
			if(len(remaining_nodes) == len(temp_list)):
				print("\tDeadLock") if(PRINT_ENABLE) else 0
				# When a deadlock occures, just keep their old values, a.k.a. do nothing, and exit the loop
				# Or maybe we should set the output nodes to 0, because otherwise a deadlock could be a good model
				for node in remaining_nodes:
					if(node.get_node_type() == Node_type.OUTPUT):
						print("\t\tSet the outputs to 0") if(PRINT_ENABLE) else 0
						node.set_value(0)
						node.set_updated(True)
				break
			# Add the nodes which we cannot calculate to the temp list
			remaining_nodes = temp_list

		# Check dictionary
		# for key in w_and_a_dict:
		# 	print(key, w_and_a_dict[key]) if(PRINT_ENABLE) else 0

		pass
