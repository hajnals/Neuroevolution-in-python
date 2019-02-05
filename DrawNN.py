from Genome import Genome
from random import uniform
from random import randint
from Genome import Node_type
import matplotlib.pyplot as plt

SIZEX = 8
SIZEY = 8
RADIUS = 0.010
SPACE = 0.10


class DrawNN:
	def __init__(self, genome):

		self.nodes 			= genome.get_node_genes()
		self.connections 	= genome.get_connection_genes()
		self.figure 		= plt.figure(figsize=(SIZEX,SIZEY))
		self.ax 			= self.figure.gca()
		self.niecheID		= genome.niecheID
		self.fitness		= genome.get_fitness()

		self.nodeCoordToID =  dict()

		plt.axis('off')

		self.drawNodes()
		self.drawConnections()
		self.drawInformation()

		plt.savefig('Fitness/genome'+str(genome.niecheID)+str(genome.get_genome_id())+'.png', bbox_inches='tight')
		plt.close()

	def drawInformation(self):
		x = 0.05
		y = 0.05
		niecheIDstr = plt.text(x,y,"Nieche:"+str(self.niecheID))

		x = 0.5
		y = 0.05
		fitnessstr = plt.text(x,y,"Fitness:"+str(self.fitness))
		
		self.ax.add_artist(niecheIDstr)
		self.ax.add_artist(fitnessstr)

	def drawConnections(self):
		for connection in self.connections:

			# If this connection is disabled continue with others
			if(connection.expressed == False):
				continue

			[nodeIn, nodeOut] = connection.get_connected_nodes_id()

			coordIn = self.getCoordOfNode(nodeIn)
			coordOut = self.getCoordOfNode(nodeOut)

			weight = connection.get_weight()

			# We cannot represent weights higher than 1 or -1, Later i can think of a solution
			if(weight > 1): weight = 1
			if(weight < -1): weight = -1

			if(weight < 0):
				colour = [1,0,0,abs(weight)]
			elif(weight > 0):
				colour = [0,1,0,abs(weight)]
			else:
				colour = [0,0,0]

			self.drawLine(coordIn[0],coordIn[1],coordOut[0],coordOut[1], colour)

	def drawLine(self, x1,y1,x2,y2, color):
		# TODO If we recursively connected ourselves make a curvly line to be it more visible
		#if()

		# Connection line with weight representation
		line = plt.Line2D([x1,x2],[y1+RADIUS,y2-RADIUS], color=color)
		self.ax.add_artist(line)

		# Slim black line to see which noeds are connected
		line_slim_black = plt.Line2D([x1,x2],[y1+RADIUS,y2-RADIUS], color=[0,0,0], linewidth=0.5)
		self.ax.add_artist(line_slim_black)

	def drawCircle(self, x, y, nodeType, nodeID):

		if(nodeType == Node_type.INPUT):
			rgb = [77, 166, 255]
			circle = plt.Circle((x,y), radius = RADIUS, fill=True, fc=(rgb[0]/255, rgb[1]/255, rgb[2]/255), ec=(0,0,0))
		elif(nodeType == Node_type.HIDDEN):
			rgb = [166, 255, 77]
			circle = plt.Circle((x,y), radius = RADIUS, fill=True, fc=(rgb[0]/255, rgb[1]/255, rgb[2]/255), ec=(0,0,0))
		elif(nodeType == Node_type.OUTPUT):
			rgb = [255, 166, 77]
			circle = plt.Circle((x,y), radius = RADIUS, fill=True, fc=(rgb[0]/255, rgb[1]/255, rgb[2]/255), ec=(0,0,0))

		x += 0.0125
		y -= 0.0075
		text = plt.text(x,y,str(nodeID))
		self.ax.add_artist(text)

		self.ax.add_artist(circle)

	def drawNodes(self):
		outputCounter = 0
		inputCounter = 0
		for node in self.nodes:
			if(node.get_node_type() == Node_type.INPUT):
				xCoord = 0+(RADIUS+SPACE)+(inputCounter*2*(RADIUS+SPACE))
				yCoord = 0+(RADIUS+SPACE)
				self.drawCircle(xCoord, yCoord, node.get_node_type(), node.get_node_id())
				self.nodeCoordToID[node.get_node_id()] = [xCoord,yCoord]
				inputCounter += 1

			elif(node.get_node_type() == Node_type.OUTPUT):
				xCoord = 0+(RADIUS+SPACE)+(outputCounter*2*(RADIUS+SPACE))
				yCoord = 1-(RADIUS+SPACE)
				self.drawCircle(xCoord, yCoord, node.get_node_type(), node.get_node_id())
				self.nodeCoordToID[node.get_node_id()] = [xCoord,yCoord]
				outputCounter += 1

			elif(node.get_node_type() == Node_type.HIDDEN):
				[xCoord, yCoord] = self.getHiddenCoords2(node.get_node_id())
				self.drawCircle(xCoord, yCoord, node.get_node_type(), node.get_node_id())
				self.nodeCoordToID[node.get_node_id()] = [xCoord,yCoord]

	# Finds an optimal place for a hidden node
	def getHiddenCoords(self, nodeID):
		coordX = 0
		coordY = 0

		#search connection of hidden node
		N = 0
		for connection in self.connections:
			tmpCoordX = 0
			tmpCoordY = 0

			[inNode, outNode] = connection.get_connected_nodes()

			if(inNode.get_node_id() == nodeID):
				[tmpCoordX, tmpCoordY] = self.getCoordOfNode(outNode.get_node_id())
			elif(outNode.get_node_id() == nodeID):
				[tmpCoordX, tmpCoordY] = self.getCoordOfNode(inNode.get_node_id())
			else:
				continue
	
			coordX += tmpCoordX
			coordY += tmpCoordY

			if(tmpCoordX == 0 and tmpCoordY == 0):
				pass
			else:
				N += 1

		if(N == 0): N=1
		coordX = coordX/N
		coordY = coordY/N

		coordX,coordY = self.check_place(coordX,coordY)

		return [coordX,coordY]

	def getHiddenCoords2(self, fake):
		coordX = round(uniform(0.2,0.8),2)
		coordY = round(uniform(0.2,0.8),2)
		coordX,coordY = self.check_place(coordX,coordY)
		return [coordX,coordY]

	# Returns the coordinates of the nodes
	def getCoordOfNode(self, nodeID):
		if nodeID in self.nodeCoordToID:
			return self.nodeCoordToID[nodeID]
		else:
			return [0,0]

	# Recursive function to find a place for the node, if the place is already allocated
	def check_place(self,coordX,coordY):
		# TODO: If there is another hidden here, try to move it a little to left or right.
		# Get the closes node, if the closes node is within 3 radius, 
		# then get another shift and continue until there isnt a node within 3 radius
		for key in self.nodeCoordToID:
			
			otherX, otherY = self.nodeCoordToID[key]

			disX = pow(otherX - coordX,2)
			disY = pow(otherY - coordY,2)

			dis = pow(disX+disY, 0.5)

			if(dis < 5*RADIUS):
				coordX += 5*RADIUS
				coordX, coordY = self.check_place(coordX,coordY)
		
		return coordX,coordY
