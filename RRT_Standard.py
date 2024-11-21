"""
Simple RRT Implementation
"""

"""
General Gist:
Pick point
Find closest vertex (Bias?)
Add step vector in direction
Repeat
"""    
import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import math
import matplotlib
import numpy as np
from PIL import Image, ImageOps
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import time
import tracemalloc

matplotlib.use("TKAgg")  # change to TKAgg during testing to display graph

class RRT:
    def __init__(self):
        # initialize start and end points
        self.startingLocation = (15,85)
        self.endingLocation = (85,85)
        self.newStepSize = 10  # Step size of each vector
        # Plane Bound 
        self.lowerBoundX = 0
        self.lowerBoundY = 0
        self.upperBoundX = 100
        self.upperBoundY = 100
        # Condition
        self.totalNode = 1
        self.shortestPath = 1000000
        self.endRadius = 7
        self.targetReached = False
        # Graph
        self.graph = nx.DiGraph()
        self.graph.add_node(0, pos=self.startingLocation, parentNode=None)

        #bias
        self.bias = 0

        # Load Image
        self.np_img = np.array(ImageOps.grayscale(Image.open('Images/empty.png')))
        self.np_img = ~self.np_img
        self.np_img[self.np_img > 0] = 1
        np.save('cspace.npy', self.np_img)
        np.savetxt('cspace.txt', self.np_img)
        self.grid = np.load('cspace.npy')

        self.createGraph()

    # Insert new node
    def connectNewVector(self):
        # Get a random point and draw a step vector in that direction
        newPoint = self.getRandomPoint() 
        (originalID, originalPoint, totalLength) = self.findClosestPoint(newPoint)
        xComp = newPoint[0] - originalPoint[0]
        yComp = newPoint[1] - originalPoint[1]
        (xFinal, yFinal) = (None, None)     
        # Obstacle Check
        try:
            # Find final vector: length must be <= 1
            if (totalLength >= self.newStepSize):
                xFinal = originalPoint[0] + (xComp / totalLength) * self.newStepSize
                yFinal = originalPoint[1] + (yComp / totalLength) * self.newStepSize
            else:
                xFinal = originalPoint[0] + xComp
                yFinal = originalPoint[1] + yComp
            # Check for validity of point
            # Boundary Check 
            if ((xFinal < self.lowerBoundX) or (xFinal > self.upperBoundX) or (yFinal < self.lowerBoundY) or (yFinal > self.upperBoundY)):
                self.connectNewVector() # Reattempt
                return
            # Step Radius Bounds
            lowestPossibleX = xFinal - self.newStepSize
            lowestPossibleY = yFinal - self.newStepSize
            highestPossibleX = xFinal + self.newStepSize
            highestPossibleY = yFinal + self.newStepSize
            lowerXBound = 0 if (lowestPossibleX < 0) else math.floor(lowestPossibleX)
            upperXBound = self.upperBoundX if (highestPossibleX > self.upperBoundX) else math.ceil(highestPossibleX)
            lowerYBound = 0 if (lowestPossibleY < 0) else math.floor(lowestPossibleY)
            upperYBound = self.upperBoundY if (highestPossibleY > self.upperBoundY) else math.ceil(highestPossibleY)
            
            # Brute force every invalid pixel within step radius
            for xPixel in range (lowerXBound, upperXBound + 1):
                for yPixel in range (lowerYBound, upperYBound + 1): 
                    cPixel = self.grid[xPixel][yPixel]
                    if (cPixel == 1): # Obstacle
                        # Iterate through each side of rectangle
                        rectSides = [((xPixel, yPixel), (xPixel + 1, yPixel)),((xPixel, yPixel), (xPixel, yPixel + 1)),((xPixel + 1, yPixel), (xPixel + 1, yPixel + 1)),((xPixel, yPixel + 1), (xPixel + 1, yPixel + 1))]
                        for sideConfig in rectSides:
                            if (self.segmentIntersect(originalPoint, (xFinal, yFinal), sideConfig[0], sideConfig[1])):# Intersection exists
                                self.connectNewVector()
                                return

        except: # Same point
            self.connectNewVector()
            return

        # Check if within radius
        if (self.findDistance((xFinal, yFinal), self.endingLocation) <= self.endRadius):
            self.targetReached = True
            # Return distance
            self.shortestPath = nx.shortest_path(self.graph)

        self.graph.add_node(self.totalNode, pos=(xFinal, yFinal), parentNode = originalID)
        self.graph.add_edge(originalID, self.totalNode, weight = 1, color = 'b')
        self.totalNode += 1

    # Find random point in graph bounds
    def getRandomPoint(self):
        bias = random.randrange(0,101,1)
        if (bias >= 0 and bias <= self.bias):
           return (random.randrange(self.endingLocation[0] - 10, self.endingLocation[0] + 5, 1), random.randrange(self.endingLocation[1]-10, self.endingLocation[1] + 5, 1))
        else:
           return (random.randrange(self.lowerBoundX, self.upperBoundX, 1), random.randrange(self.lowerBoundY, self.upperBoundY, 1))

    # Find  distance between two coordinates
    def findDistance(self, coord1, coord2):
        return math.sqrt(abs(coord1[0] - coord2[0])**2 + abs(coord1[1] - coord2[1])**2)

    # Find closest point to benchmark
    def findClosestPoint(self, benchmarkPoint):
        (closestKey, closestNode) = (None, None)
        closestDistance = 10000000
        benchmarkCoord = benchmarkPoint
        pointData = nx.get_node_attributes(self.graph, "pos")
        # Iterate through all points
        for cKey in pointData:
            cPoint = pointData[cKey]
            birdseyeDistance = self.findDistance(benchmarkCoord, cPoint)
            if (birdseyeDistance < closestDistance):
                closestNode = cPoint
                closestDistance = birdseyeDistance
                closestKey = cKey
        return (closestKey, closestNode, closestDistance)

    # Checks orientation of three points representing two line segments
    def ccwOrientationCheck(self,point1,point2,point3):
        return (((point3[1]-point1[1]) * (point2[0]-point1[0])) > ((point2[1]-point1[1]) * (point3[0]-point1[0])))

    # True if new vector and imaginary side intersect
    def segmentIntersect(self,originalPoint,newPoint,rect1,rect2):
        return ((self.ccwOrientationCheck(originalPoint,rect1,rect2) != self.ccwOrientationCheck(newPoint,rect1,rect2)) and (self.ccwOrientationCheck(originalPoint,newPoint,rect1) != self.ccwOrientationCheck(originalPoint,newPoint,rect2)))

    # Creates graph of the value 
    def createGraph(self):
        # Plot that shows no obstacles
        # plt.xlim(self.lowerBoundX, self.upperBoundX)
        # plt.ylim(self.lowerBoundY, self.upperBoundY)
        # pos = nx.get_node_attributes(self.graph, "pos")
        # nx.draw(self.graph, pos, node_size=5, node_color="green", edge_color="green") #pos,
        # plt.show()

        # Load Image
        displayGrid = np.load('cspace.npy')
        fig = plt.figure('RRT')

        # Image transformations
        displayGrid = np.rot90(displayGrid)
        displayGrid = np.flip(displayGrid)
        displayGrid = np.fliplr(displayGrid)

        # RRT until connected
        maxIter = 2500
        numIter = 0

        start_time = time.time()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        while ((self.targetReached == False)and(numIter < maxIter)):
           self.connectNewVector()
           self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
           numIter += 1

        print("--- %s seconds ---" % (time.time() - start_time))
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total = 0
        for stat in top_stats:
            total += stat.size
        print("memory usage: " + str(total/1024) + " KiB")

        parentNode = nx.get_node_attributes(self.graph, 'parentNode')
        path = []
        currentNode = self.totalNode - 1
        while (parentNode[currentNode] != None):
            path.append(currentNode)
            currentNode = parentNode[currentNode]
        
        path.append(0)

        path.reverse()

        # Return shortest path
        for current, next_item in zip(path, path[1:]):
            self.graph.add_edge(current, next_item, color = 'g', weight = 4)
        shortestLength = len(path) - 1
        pointData = nx.get_node_attributes(self.graph, "pos")[path[2]]
        pointList = []
        for x in path:
            pointList.append(nx.get_node_attributes(self.graph, "pos")[x])
        print(pointList)



        def BezierCurve(self,P0,P1,P2):
            tValues = np.linspace(0,1,num=1000)
            xValues = np.array([((1-t)**2)*P0[0] + 2*(1-t)*t*P1[0] + (t**2)*P2[0] for t in tValues])
            yValues = np.array([((1-t)**2)*P0[1] + 2*(1-t)*t*P1[1] + (t**2)*P2[1] for t in tValues])

            return xValues, yValues
        
        def ContinuousBezier(self, points):
            xBezier = []
            yBezier = []
            for i in range(len(points)-2):
                if i%2 ==0:
                    P0 = points[i]
                    P1 = points[i+1]
                    P2 = points[i+2]
                    xValues, yValues = BezierCurve(self,P0,P1,P2)
                    xBezier.append(xValues)
                    yBezier.append(yValues)
            return xBezier, yBezier
        
        xBezier, yBezier = ContinuousBezier(self, pointList)
        plt.plot([p[0] for p in pointList], [p[1] for p in pointList], 'o-')
        for i in range(len(xBezier)):
            plt.plot(xBezier[i], yBezier[i])

        


        # Statuses
        print(self.graph)
        print(f"Shortest Path: {path}")
        print(f"Shortest Path Length: {shortestLength}")

       

        # Draw Graph
        plt.imshow(displayGrid, cmap = 'binary')
        plt.plot(*self.startingLocation, 'ro')
        plt.plot(*self.endingLocation,'go')
        goalRegion = plt.Circle(self.endingLocation, self.endRadius, color='b', fill = False)
        pos = nx.get_node_attributes(self.graph, "pos")

        ax = plt.gca()
        ax.set_xlim(self.lowerBoundX, self.upperBoundX)
        ax.set_ylim(self.lowerBoundY, self.upperBoundY)
        #ax.set_ylim(ax.get_ylim()[::-1])
        ax.add_patch(goalRegion)
        ax.xaxis.set_tick_params(labelbottom=True)  
        #fig.add_axes([0.2,0.2,0.6,0.6])
        
        # Animator call
        self.index = 0
        #anim = animation.FuncAnimation(fig, self.animate, frames=math.ceil(len(self.graph)/25) + 1, interval=100, repeat=False)
        colors = [self.graph[u][v]['color'] for u,v in self.graph.edges()]
        weights = [self.graph[u][v]['weight'] for u,v in self.graph.edges()]
        nx.draw(self.graph, pos, node_size=self.endRadius, node_color="blue", edge_color = colors, width = weights)
        plt.show()

    def animate(self, index):
        targets = self.graph.subgraph([-1, 0])
        H = self.graph.subgraph(list(self.graph.nodes)[0:index * 25])
        #self.index += +

        #print (index, "list: ", list(self.graph.nodes)[0:index+1], H)

        # update graph positioning
        pos=nx.get_node_attributes(H,'pos')
        pos_t=nx.get_node_attributes(targets,'pos')
        nx.draw(H, pos, node_size=15, node_color="blue", edge_color="black")
        if (index * 25 >= len(self.graph)):
            colors = [self.graph[u][v]['color'] for u,v in self.graph.edges()]
            weights = [self.graph[u][v]['weight'] for u,v in self.graph.edges()]
            nx.draw(self.graph, pos, node_size=self.endRadius, node_color="blue", edge_color = colors, width = weights) 
        nx.draw(targets, pos_t, node_size=15, node_color="red")





def main():
    print ("Running: ")
    rrt = RRT()
    print ("Finished.")

if (__name__ == "__main__"):
    main()


# Unused
# Obstacle Check (Sample code)
# testPoint = np.array([0.0,0.0])
# for i in range(0, self.newStepSize):
#     testPoint[0] = originalPoint[0] + i*(originalPoint[0] + xComp / totalLength)
#     testPoint[1] = originalPoint[1] + i*(originalPoint[1] + yComp / totalLength)
#     if (self.grid[round(testPoint[0]), round(testPoint[1])] == 1):
#         self.connectNewVector(originalPoint)


# self.np_img = np.array(ImageOps.grayscale(Image.open('square.png')))
# self.np_img = ~self.np_img # Invert image ?
# self.np_img[self.np_img > 0] = 1 # Safe
# plt.set_cmap('binary')
# #plt.imshow(self.np_img)
# np.save('cspace.npy', self.np_img)