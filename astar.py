import numpy as np
from PIL import Image, ImageOps
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():
    graph = nx.Graph()
    
    # = plt.figure('A*')
    
    np_img = np.array(ImageOps.grayscale(Image.open('complex.png')))
    np_img = ~np_img
    np_img[np_img > 0] = 1
    np.save('cspace.npy', np_img)
    grid = np.load('cspace.npy')

    displayGrid = np.load('cspace.npy')
    displayGrid = np.rot90(displayGrid)
    displayGrid = np.flip(displayGrid)
    displayGrid = np.fliplr(displayGrid)
    plt.imshow(displayGrid, cmap = 'binary')

    

    start = (10, 10)
    end = (85, 85)
    start_time = time.time()
    path = astar(grid, start, end)
    print("--- %s seconds ---" % (time.time() - start_time))
    nodeNum = 0
    graph.add_node(nodeNum, pos = path[0])
    nodeNum += 1
    edgeNum = 0
    for nodePos in path:
        graph.add_node(nodeNum, pos = nodePos)
        nodeNum += 1

    for i in range(len(path)):
        graph.add_edge(i, i+1)
        # temp = nodeNum - 1
        # graph.add_edge(temp, nodeNum)
        # edgeNum += 1


    #graph.add_edges_from(path)

    pos=nx.get_node_attributes(graph,'pos')
    nx.draw(graph, pos, node_size=3, node_color="blue", edge_color="black")
    #nx.draw(graph)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

    print(path)


if __name__ == '__main__':
    main()