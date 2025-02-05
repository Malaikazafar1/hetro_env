import numpy as np
import heapq
import copy
import pickle
from functions import *
from classes_correction import *
import random
import time

value_of_arranging = 0

with open("myobject_3D_diagonal.pkl", "rb") as file:
    set_of_nodes = pickle.load(file)

obstacles_nodes = []
reference_dict = set_of_nodes[3]
nodes_to_load = set_of_nodes[1]
for i in obstacles_nodes:
    nodes_to_load[reference_dict[i]].obstacle_node = True

open_list = []

locations = [(0.5, 0.5, 0.5),
  (0.5, 1.5, 0.5),
  (0.5, 2.5, 0.5),
  (1.5, 2.5, 0.5),
  (1.5, 1.5, 0.5),
  (1.5, 0.5, 0.5),
  (2.5, 0.5, 0.5),
  (2.5, 1.5, 0.5),
  (2.5, 2.5, 0.5),
  (2.5, 2.5, 1.5),
  (2.5, 1.5, 1.5),
  (2.5, 0.5, 1.5),
  (1.5, 0.5, 1.5),
  (1.5, 1.5, 1.5),
  (1.5, 2.5, 1.5),
  (0.5, 2.5, 1.5),
  (0.5, 1.5, 1.5),
  (0.5, 0.5, 1.5),
  (0.5, 0.5, 2.5),
  (0.5, 1.5, 2.5),
  (0.5, 2.5, 2.5),
  (1.5, 2.5, 2.5),
  (1.5, 1.5, 2.5),
  (1.5, 0.5, 2.5),
  (2.5, 0.5, 2.5),
  (2.5, 1.5, 2.5),
  (2.5, 2.5, 2.5)]


def get_random_locations(locations, n):
    if n > len(locations):
        raise ValueError("n cannot be greater than the number of available locations")
    return random.sample(locations, n)

# Example usage
n = 4 # Change this value to select a different number of locations
random_locations = get_random_locations(locations, n)
print(random_locations)
random_locations_2 = copy.deepcopy(random_locations)
random_locations_2.reverse()
random_locations.extend(random_locations_2)
locations = random_locations
# locations = [(0.5 ,0.5, 0.5),(2.5 ,2.5, 2.5),(2.5 ,2.5, 2.5),(0.5 ,0.5, 0.5),(0.5, 2.5, 0.5), (1.5, 2.5, 1.5)]
agent_list = [0]
while locations != []:
    agent_list.append(
        agents(
            set_of_nodes[1][reference_dict[locations.pop()]],
            set_of_nodes[1][reference_dict[locations.pop()]],
        )
    )

high_level = high_level_node(
    nodes_to_load,
    agents=agent_list,
    reference_dict=reference_dict,
    reference_nodes=nodes_to_load,
    timestep_start_dict={
        i: (0, agent_list[i].start_node.location) for i in range(1, len(agent_list))
    },
)

k = []
t = 0

for i in range(1, len(high_level.agents)):
    a,b = solution = high_level.find_path_3_D_diagonal(
        agent_num=i,
        start_node=high_level.agents[i].start_node,
        dest_node=high_level.agents[i].goal_node,
    )
    k.append(a)
    # print(b_2)
   
    high_level.node_cost.append(b)
    if b > t:
        t = b

k_2 = []
for i in k:
    _ = np.pad(i, (0, t + 1 - len(i)), mode="constant", constant_values=(0, i[-1]))
    k_2.append(_)

k_2 = np.array(k_2).T
high_level.node_grid = k_2
high_level.path_cost = np.sum(high_level.node_cost)
high_level.timestep = 0
heapq.heappush(open_list, (high_level.path_cost, id(high_level), high_level))
j = 0
print('added the node')
while open_list != []:
    print('Length of open list = '+str(open_list))
    time.sleep(1)
    path_cost, i_d, present_node = heapq.heappop(open_list)
    final_time = time.time()

    j += 1
    # print('chec before checking conflicts = ' + str(present_node.nodes[28].explored))
    a = present_node.check_conflicts()
    print('reached here')
    # print('chec after checking conflicts = ' + str(present_node.nodes[28].explored))
    if a == None:
        present_node.output = present_node.node_grid
        print("The solution was reached")
        break

    print(a)
    a, b, c = a
    # print([[f.location for f in f_2] for f_2 in present_node.node_grid])
    # present_node_dict[present_node.timestep].append([[f.location for f in f_2] for f_2 in present_node.node_grid])
    if type(a) != tuple:
        constraints = [(b[0], c, a), (b[1], c, a)]
        # print(str([(x,y,z.location) for (x,y,z) in constraints]) +'vertex')
    else:
        constraints = [(a, b[0], c), ((a[1], a[0]), b[1], c)]
        # print(str([((x[0].location,x[1].location),y,z) for (x,y,z) in constraints]) + 'edge')

    node_grid = present_node.node_grid
    array = [[], []]
    for i in node_grid:
        array[0].append(copy.copy(i))
        array[1].append(copy.copy(i))

    array[0] = np.array(array[0])
    array[1] = np.array(array[1])

    nodes_for_children = [copy.deepcopy(present_node.nodes) for i in range(2)]
    # print('in nodes for children = '+ str([x[28].explored for x in nodes_for_children]))
    for i in range(2):

        next_node = high_level_node(
            nodes=nodes_for_children.pop(),
            agents=agent_list,
            conflicts=constraints[i],
            node_grid=array[i],
            reference_dict=reference_dict,
            reference_nodes=nodes_to_load,
            timestep=0,
        )  # high_level_node(nodes_for_children.pop(),agents = agent_list,conflicts = constraints[i], node_grid = array[i],reference_dict = reference_dict,reference_nodes = nodes_to_load)
        next_node.predecessor = present_node
        next_node.path_cost = present_node.path_cost
        next_node.node_cost = copy.copy(present_node.node_cost)
        next_node.path_complete_dict = copy.copy(present_node.path_complete_dict)
        next_node.timestep_start_dict = copy.copy(present_node.timestep_start_dict)
        next_node.timestep_dict = copy.copy(present_node.timestep_dict)

        if type(next_node.conflict[0]) != tuple:
            agent = next_node.conflict[0]
            next_node.nodes[
                reference_dict[next_node.conflict[2].location]
            ].constraints.append((next_node.conflict[0], next_node.conflict[1]))
        else:
            agent = next_node.conflict[1]
            next_node.nodes[
                reference_dict[next_node.conflict[0][1].location]
            ].constraints.append(
                (
                    next_node.conflict[1],
                    next_node.conflict[0][0].location,
                    next_node.conflict[2],
                )
            )

            no_solution = False
            value_of_arranging = 0
            print('oops')

        if present_node.timestep + c > next_node.node_cost[agent]:
            continue

        if value_of_arranging == 1:
            print("This was the breaking point")
            print("Found the error")
            break

        the_solution = next_node.find_path_3_D_diagonal(
            agent_num=agent,
            start_node=next_node.reference_nodes[
                next_node.reference_dict[next_node.timestep_start_dict[agent][1]]
            ],
            dest_node=next_node.agents[agent].goal_node,
            timestep=next_node.timestep_start_dict[agent][0],
        )
        if the_solution == None:
            print("None")
            continue
        # b_2 = the_solution[2]
        # the_solution = the_solution[:2]
        # print('present_node.timestep and c = ' + str(present_node.timestep) + ' ' + str(c))
        # print('The solution[0] = '+ str([e.location for e in the_solution[0]]))
        value_of_arranging = next_node.arrange_node_grid(
            returned_path=the_solution[0], agent_num=agent
        )
        if value_of_arranging == 1:
            print("The value of arranging = " + str(value_of_arranging))
        # print([[f.location for f in f_2] for f_2 in next_node.node_grid])
        if value_of_arranging == 1:

            print("This was the breaking point")
            break

        next_node.node_cost[agent] = the_solution[1]
        next_node.path_cost = np.sum(next_node.node_cost)
        next_node.path_complete_dict[agent] = False
        heapq.heappush(open_list, (next_node.path_cost, id(next_node), next_node))
       

if a == None:

    final_array = [[x.location for x in i] for i in present_node.node_grid]
    print("final_array = " + str(final_array))
    final_array = np.array(final_array)
    final_array = final_array.T
    # print(final_array)
    truth_values = [
        list(zip(final_array[0][i], final_array[1][i],final_array[2][i]))
        for i in range(len(final_array[0]))
    ]
    print(truth_values)
    truth_values_2 = [
        [
            (
                True
                if x == 0
                or (
                    nodes_to_load[reference_dict[i[x]]]
                    in nodes_to_load[reference_dict[i[x - 1]]].osf_dict.values()
                    or i[x - 1] == i[x]
                )
                else False
            )
            for x in range(len(i))
        ]
        for i in truth_values
    ]
    print(truth_values_2)


# print("secondary conflicts = " + str(present_node.secondary_conflict))
# print("vertex conflicts = " + str(present_node.vertex_conflicts))   

