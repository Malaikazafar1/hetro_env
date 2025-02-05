from functions import *
import numpy as np
import heapq
import copy
import time as tim

square_root_three_half = np.sqrt(3)/2
square_root_three = np.sqrt(3)
node_dict = {'explored': False, 'predecessor':0,'timestep':0,'f_list':[],'f_static':0,'f_cost':0,'f_dict':{},'prev_node_code':0,'original_timestep':0,'wait': False,'imaginary_successor':[],'my_number':10,'next': 0}
movement_dict = {'movement_tuple': 0, 'waiting': False,'waiting_agent':0}

class clusters():
    def __init__(self,number):

        self.cluster_for_agents = [number]
        self.number_of_collisisons = 0

class agents():
    def __init__(self,start_node,goal_node,path_cost=0,the_number = 0):
        
        self.start_node  = start_node
        self.goal_node = goal_node
        self.timestep = 0
        self.collision_number =  0
        self.anchor_agent = the_number

class low_level_node():
    def __init__(self,location = (),actions = list(range(1,10))):
        
        self.f_static = 0
        self.g_static = 0
        self.f_cost = 0
        self.location = location
        self.actions = actions
        self.osf_dict = {i:0 for i in actions}
        self.f_list = []
        self.explored = False
        self.predecessor = 0  
        self.obstacle_node = False
        self.timestep = 0
        self.constraints = []
        self.dummy = False
        self.wait = True
        self.multidimensional_explore = {0:copy.deepcopy(node_dict)}
        
class low_level_node_hexagon():
    def __init__(self,location = (),actions = list(range(1,7))):
        
        self.f_static = 0
        self.g_static = 0
        self.f_cost = 0
        self.location = location
        self.actions = actions
        self.osf_dict = {i:0 for i in actions}
        self.f_list = []
        self.explored = False
        self.predecessor = [] 
        self.obstacle_node = False
        self.timestep = 0
        self.constraints = []
        self.waiting_constraints = []
        self.turning_constraints = []
        self.dummy = False
        self.wait = True
        self.agent_list = []
        self.imaginary_agent_list = []
        self.imaginary_successor = 0
        self.successor = []
        self.multidimensional_explore = {0:copy.deepcopy(node_dict)}
        
class high_level_node():
    def __init__(self,nodes,agents,conflicts = [[[]],[[]]],node_grid = [],reference_dict = {},reference_nodes = [],timestep = 0,timestep_dict = {},timestep_start_dict = {}):
       
        self.timestep_dict = timestep_dict
        self.output = []
        self.path_complete_dict = {i:False for i in range(1,len(agents))}
        self.nodes = nodes
        self.agents = agents
        self.conflict  = conflicts
        self.timestep = timestep
        self.node_grid = node_grid
        self.dummy = False
        self.predecessor = 0
        self.node_cost = [0]
        self.path_cost = 0
        self.reference_dict = reference_dict
        self.timestep_start_dict = timestep_start_dict
        self.reference_nodes = reference_nodes
        self.secondary_conflict = {}
        self.vertex_conflicts = {}
        self.false_node_grid = []
        self.imaginary_agent_timesteps = {}
        self.agent_positions = {}
        self.future_positions = {}
        self.successor_agent_timesteps = {}
    #find_distance = lambda x,y,z: np.linalg.norm(np.array(list(y)) - np.array(list(z))) - np.linalg.norm(np.array(list(x)) - np.array(list(z))         
   
    def find_path_2(self,final_node,initial_node):
        'we did reach here'
        node_list = []
        x = final_node
        node_list.append(x)
        while x != initial_node:
            x = x.predecessor
            if x.dummy:
                print('dummy found')
                t = 0
                while x.dummy:
                    t += 1
                    x = x.predecessor
                
                node_list.extend([x for i in range(t)])      
            node_list.append(x)

        time = final_node.timestep
        node_list.reverse()
        node_list = [self.reference_nodes[self.reference_dict[x.location]] for x in node_list]
        if False in [True if x == 0 or (node_list[x] in node_list[x-1].osf_dict.values() or node_list[x] == node_list[x-1]) else False for x in range(len(node_list))]:
            print('Flawed_node_list = ' + str([x.location for x in node_list]))
            return None
        return (node_list,time)
    
    def find_path_array_colregs(self,final_node,initial_node,stop_count = 0):
        imaginary_tuple  = []
        successor_tuple = []
        node_list = []
        x = final_node
        #stop_count = list(final_node.multidimensional_explore.keys())[-1]
        time = final_node.multidimensional_explore[stop_count]['timestep']
        if stop_count != 0:
            print('stop_count = ' + str(stop_count))
           # tim.sleep(2)
        node_list.append(x)
        while (x != initial_node or stop_count != 0):
            previous_node = x
            # print(x)
            # print(x == initial_node)
            # print('dict = ' + str(x.multidimensional_explore))
            # print(x.multidimensional_explore[stop_count])
            # print('stop_count = ' + str(stop_count))
            x = x.multidimensional_explore[stop_count]['predecessor']
            if x.multidimensional_explore[stop_count]['imaginary_successor'] != []:
                for _ in x.multidimensional_explore[stop_count]['imaginary_successor']:
                    imaginary_tuple.append((_,x.multidimensional_explore[stop_count]['original_timestep'] + 1))
            #print(x.multidimensional_explore[stop_count])
            if x.multidimensional_explore[stop_count]['original_timestep'] != x.multidimensional_explore[stop_count]['timestep']:
                if node_list[-1].location != x.location:
                    successor_tuple.append((node_list[-1].location,x.multidimensional_explore[stop_count]['original_timestep']))
            node_list.append(x)
            if x == previous_node:
                stop_count -= 1
                #x = self.reference_nodes[self.reference_dict[x.multidimensional_explore[stop_count]['predecessor']]]
               # node_list.extend([x for i in range(2)])
          
        node_list.reverse()
        node_list = [self.reference_nodes[self.reference_dict[x.location]] for x in node_list]
        imaginary_tuple_2 = []
        for _ in imaginary_tuple:
            if node_list[_[1]].location != _[0]:
                imaginary_tuple_2.append(_)
        
        if False in [True if x == 0 or (node_list[x] in node_list[x-1].osf_dict.values() or node_list[x] == node_list[x-1]) else False for x in range(len(node_list))]:
            print('Flawed_node_list = ' + str([x.location for x in node_list]))
            return None
        return (node_list,time,imaginary_tuple_2,successor_tuple)
    
    def find_path_array_3D(self,final_node,initial_node,stop_count = 0):
        
        imaginary_tuple  = []
        successor_tuple = []
        node_list = []
        x = final_node
        #stop_count = list(final_node.multidimensional_explore.keys())[-1]
        time = final_node.multidimensional_explore[stop_count]['timestep']
        if stop_count != 0:
            print('stop_count = ' + str(stop_count))
           # tim.sleep(2)
        node_list.append(x)
        while (x != initial_node or stop_count != 0):
            previous_node = x
            x = x.multidimensional_explore[stop_count]['predecessor']
            stop_count -= 1
            # if x.multidimensional_explore[stop_count]['original_timestep'] != x.multidimensional_explore[stop_count]['timestep']:
            #     if node_list[-1].location != x.location:
            #         successor_tuple.append((node_list[-1].location,x.multidimensional_explore[stop_count]['original_timestep']))
            node_list.append(x)
            # if x == previous_node:
            #     stop_count -= 1
                #x = self.reference_nodes[self.reference_dict[x.multidimensional_explore[stop_count]['predecessor']]]
               # node_list.extend([x for i in range(2)])
          
        node_list.reverse()
        node_list = [self.reference_nodes[self.reference_dict[x.location]] for x in node_list]
        # imaginary_tuple_2 = []
        # for _ in imaginary_tuple:
        #     if node_list[_[1]].location != _[0]:
        #         imaginary_tuple_2.append(_)
        
        # if False in [True if x == 0 or (node_list[x] in node_list[x-1].osf_dict.values() or node_list[x] == node_list[x-1]) else False for x in range(len(node_list))]:
        #     print('Flawed_node_list = ' + str([x.location for x in node_list]))
        #     return None
        return (node_list,time)
    
    def find_path_2D(self,agent_num,constraints = [],start_node = 1,dest_node = 2,timestep = 0):
        
        low_level_nodes = 0 
        set_of_nodes = copy.deepcopy(self.nodes)
        start_node = set_of_nodes[self.reference_dict[start_node.location]]
        dest_node = set_of_nodes[self.reference_dict[dest_node.location]]
        again = False
        open = []
        a = np.max((np.abs(start_node.location[0] - dest_node.location[0]),np.abs(start_node.location[1] - dest_node.location[1])))
        heapq.heappush(open,(a,id(start_node),0,start_node)) 
        low_level_nodes += 1
        closed = []
        indexerror = node_suppressed = just_created = new_node_creation = False
        start_node.timestep = timestep 
        start_node.multidimensional_explore[0] = copy.deepcopy(node_dict)
        start_node.multidimensional_explore[0]['timestep'] = start_node.multidimensional_explore[0]['original_timestep'] = timestep
        
        output = 0
        no_node_added = False
        counter = 0
        while open != []:
            
            reached_here = False
            if no_node_added:
                no_node_added = False
                
            else:                                              ## Node ran out of neighbours due to constraints
                x,i_d,stop_count,present_node = heapq.heappop(open)
                counter += 1

               
                    
            if present_node.f_list == [] :
                
                a,b = (abs(-present_node.location[0] + dest_node.location[0]), abs(- present_node.location[1] + dest_node.location[1]))

                present_node.f_static = x       # Chebychev distance
                g = lambda x: 0 if x == 0 else (1 if x > 0 else -1)
                direction = [g(a),g(b)]
                if np.abs(a) == np.abs(b):
                    h_values = osf_hexagon(direction,2)
                
                else:
                    c = np.abs(a) - np.abs(b)
                    h_values = osf_hexagon(direction = direction,greater_direction = (lambda x : 0 if x > 0 else 1)(c), diff_one = True if abs(c) < 3/2 + square_root_three_half else False, direction_diff = abs(c))
                f_values = list(enumerate([square_root_three + h_values[i] for i in range(6)],start = 1))
                                                                        # Don't forget to change the value for key 9 in class node dictionary.
                f_values_2 = sorted(np.unique([i[1] for i in f_values]),reverse = True)
                present_node.f_dict = {i:[j[0] for j in f_values if j[1] == i and present_node.osf_dict[j[0]]!= 0] for i in f_values_2}
                #print(present_node.f_dict)
                present_node.f_list = f_values_2
                #print(present_node.f_list)  
                present_node.f_cost = present_node.f_static + present_node.f_list[-1]
                if (agent_num,present_node.timestep+1) in present_node.constraints:
                    present_node.wait = False

                present_node.multidimensional_explore[stop_count]['f_list'] = copy.deepcopy(present_node.f_list)
                present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.f_list[-1] 
                present_node.multidimensional_explore[stop_count]['f_dict'] = copy.deepcopy(present_node.f_dict)
                                
            next_nodes = []                              ## Edit over here as I was restricting the motion to access further nodes by making it wait at a particular timestep.
            while next_nodes == []:                                     ## How 

                try:

                    change_in_f =  present_node.multidimensional_explore[stop_count]['f_list'].pop()   
                  
                except IndexError:
                    indexerror = True
                    break
                if dest_node.explored:
                    print('dest_node explored')
                
                for i in present_node.osf_dict:
                    if present_node.osf_dict[i] != 0 :
                        if stop_count + 1 not in present_node.osf_dict[i].multidimensional_explore:
                            present_node.osf_dict[i].multidimensional_explore[stop_count + 1] = copy.deepcopy(node_dict)
                            # print('successor = ' + str(present_node.osf_dict[i].multidimensional_explore[stop_count]['next']))
                next_nodes = [x for x in present_node.multidimensional_explore[stop_count]['f_dict'][change_in_f] if not (present_node.osf_dict[x].obstacle_node or present_node.osf_dict[x].multidimensional_explore[stop_count + 1]['explored'])]
 

                if again:
                    print('next_nodes = ' + str(next_nodes))
                           
            if indexerror:  
                indexerror = False
                continue
           
            # if present_node.multidimensional_explore[stop_count]['original_timestep'] != present_node.multidimensional_explore[stop_count]['timestep']:
            #     if [x for x in next_nodes if (agent_num, present_node.location, present_node.multidimensional_explore[stop_count]['timestep'] + 1) in present_node.osf_dict[x].waiting_constraints] != []:
            #         node_suppressed = True
            #         next_nodes = [x for x in next_nodes if (agent_num ,present_node.location, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].waiting_constraints]

            if [x for x in next_nodes if (agent_num , present_node.location , present_node.multidimensional_explore[stop_count]['timestep'] + 1) or (agent_num,present_node.multidimensional_explore[stop_count]['timestep'] + 1) in present_node.osf_dict[x].constraints] != []:
                node_suppressed = True
                next_nodes = [x for x in next_nodes if (agent_num ,present_node.location ,present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].constraints and (agent_num,present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].constraints]   ## Egge Constraints


        
            if again:

                print('next_nodes again = ' + str(next_nodes))
                print(present_node.multidimensional_explore[stop_count]['f_dict'])
                time.sleep(2)
            
            if node_suppressed or change_in_f == present_node.f_list[0] or change_in_f >= 1:
                    if again:
                        print('reached here')
                        time.sleep(1)
                    if (agent_num, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.constraints and (not present_node.multidimensional_explore[stop_count]['wait']) :
                        new_node_creation = True
                            
            if new_node_creation:
                node_suppressed = False
                new_node_creation = False
            
                if (agent_num, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.constraints:
                    node_already_present = False
                    if stop_count + 1 in present_node.multidimensional_explore:
                        if present_node.multidimensional_explore[stop_count + 1]['explored']:
                            node_already_present = True
                            if again:
                                print('instead we had to jump here')
                                time.sleep(1)
                             
                    if not node_already_present:
                        just_created = True
                        if again:
                            print('we did reach here')

                            time.sleep(1)
                        present_node.multidimensional_explore[stop_count + 1] = copy.deepcopy(node_dict)
                      #  print('successor = ' + str(present_node.multidimensional_explore[stop_count + 1]['next']))
                        # if present_node.multidimensional_explore[stop_count]['original_timestep'] != 0 :
                        #     present_node.multidimensional_explore[stop_count + 1]['my_number'] = present_node.multidimensional_explore[stop_count]['my_number']

                        present_node.multidimensional_explore[stop_count + 1]['f_list'] = copy.deepcopy(present_node.f_list)
                        present_node.multidimensional_explore[stop_count + 1]['f_dict'] = copy.deepcopy(present_node.multidimensional_explore[stop_count]['f_dict'])
                    # new_node.dummy = True
                        #new_node.explored = True
                        present_node.multidimensional_explore[stop_count + 1]['explored'] = True
                        #new_node.timestep = present_node.timestep + 1
                        present_node.multidimensional_explore[stop_count + 1]['timestep'] = present_node.multidimensional_explore[stop_count]['timestep'] + 1
                        #new_node.f_static = present_node.f_static + 1
                        present_node.multidimensional_explore[stop_count + 1]['original_timestep'] =  present_node.multidimensional_explore[stop_count]['original_timestep']
                        present_node.multidimensional_explore[stop_count + 1]['f_static'] = present_node.multidimensional_explore[stop_count]['f_static'] + 1
                        #new_node.f_cost = change_in_f + present_node.f_static
                        present_node.multidimensional_explore[stop_count + 1]['f_cost'] = present_node.multidimensional_explore[stop_count + 1]['f_list'][-1] + present_node.multidimensional_explore[stop_count + 1]['f_static']
                        #new_node.predecessor = present_node
                        present_node.multidimensional_explore[stop_count + 1]['predecessor'] = present_node
                        present_node.multidimensional_explore[stop_count]['wait'] = True
                        low_level_nodes += 1

                        heapq.heappush(open,(present_node.multidimensional_explore[stop_count + 1]['f_static'],id(present_node),stop_count + 1,present_node))
                else:
                    if again:
                        print('we are here')
                        time.sleep(1)
            if (not just_created) and next_nodes == []:
                if len(open) != 0:
                    if present_node.multidimensional_explore[stop_count]['f_list'] !=[]:
                        no_node_added = True
                        continue
            
            just_created = False
            if present_node.multidimensional_explore[stop_count]['f_list'] != []:
                present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.multidimensional_explore[stop_count]['f_list'][-1]
                heapq.heappush(open,(present_node.multidimensional_explore[stop_count]['f_cost'],id(present_node),stop_count,present_node))
            
            if next_nodes != []:
 
                for i in next_nodes:
                    low_level_nodes += 1
                    #present_node.osf_dict[i].explored = True
                    # present_node.osf_dict[i].multidimensional_explore[stop_count]['my_number'] = i
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['explored'] = True
                    #present_node.osf_dict[i].predecessor = present_node
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['predecessor'] = present_node
                    #present_node.osf_dict[i].timestep = present_node.timestep + 1
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['timestep'] = present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['original_timestep'] = present_node.multidimensional_explore[stop_count]['timestep'] + 1
                    if present_node.osf_dict[i].f_list != []:
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_list'] = copy.deepcopy(present_node.osf_dict[i].f_list)
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_dict'] = copy.deepcopy(present_node.osf_dict[i].f_dict)
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_static'] = present_node.multidimensional_explore[stop_count]['f_static'] + change_in_f
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_cost'] = present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_static'] + present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['f_list'][-1]

                    heapq.heappush(open,(present_node.multidimensional_explore[stop_count]['f_static'] + change_in_f,id(present_node.osf_dict[i]),stop_count + 1,present_node.osf_dict[i]))
                    #print('locations of present_node and the suggested node = ' + str(present_node.location) + ' ' + str(present_node.osf_dict[i].location))
                    if present_node.osf_dict[i] == dest_node:
                        output = 1
                        break
            
            if output == 1:
                break
        
        if output == 1 :
            
            x_1,y_1 =  self.find_path_array_3D(final_node = dest_node,initial_node = start_node,stop_count = stop_count + 1) 
            
            return (x_1,y_1)        
        else:
            return None
          
    def find_path_3_D_diagonal(self,agent_num,start_node = 1,dest_node = 2,timestep = 0, again = False):
        
        low_level_nodes = 0 
        set_of_nodes = copy.deepcopy(self.nodes)
        start_node = set_of_nodes[self.reference_dict[start_node.location]]
        dest_node = set_of_nodes[self.reference_dict[dest_node.location]]
        
        open = []
        a = np.max((np.abs(start_node.location[0] - dest_node.location[0]),np.abs(start_node.location[1] - dest_node.location[1])))
        heapq.heappush(open,(a,id(start_node),0,start_node)) 
        low_level_nodes += 1
        closed = []
        indexerror = node_suppressed = just_created = new_node_creation = False
        start_node.timestep = timestep 
        start_node.multidimensional_explore[0] = copy.deepcopy(node_dict)
        start_node.multidimensional_explore[0]['timestep'] = start_node.multidimensional_explore[0]['original_timestep'] = timestep
        
        print(start_node.location)
        print(dest_node.location)
        output = 0
        no_node_added = False
        counter = 0
        while open != []:
            reached_here = False
            if no_node_added:
                no_node_added = False
                
            else:                                              ## Node ran out of neighbours due to constraints
                x,i_d,stop_count,present_node = heapq.heappop(open)
                counter += 1
                print(start_node.location)
                print(dest_node.location)
                print(present_node.location)
                print(stop_count)
                tim.sleep(2)
      
            if present_node.f_list == [] :
                
                a,b,c = (abs(-present_node.location[0] + dest_node.location[0]), abs(- present_node.location[1] + dest_node.location[1]), abs(- present_node.location[2] + dest_node.location[2]))
                greater_value = max(a,b,c)

                c_ = 1
                coordinates = [(0, c,0),(0,-c,0),(c,0,0),(-c,0,0),(0,0,-c),(0,0,c),(c,c,c),(c,-c,c),(-c,-c,c),(-c,-c,-c),(c,-c,-c),(c,c,-c),(-c,c,-c),(-c,c,c)]
                greatest_values_neighbours = np.array([(0, c,0),(0,-c,0),(c,0,0),(-c,0,0),(0,0,-c),(0,0,c),(c,c,c),(c,-c,c),(-c,-c,c),(-c,-c,-c),(c,-c,-c),(c,c,-c),(-c,c,-c),(-c,c,c)]) + present_node.location
                h_values_pre = np.array([max(tuple(i)) for i in list(np.abs(dest_node.location - greatest_values_neighbours))])
                h_values = list(h_values_pre - greater_value)
                f_values = list(enumerate([round(np.sqrt(np.sum(np.array(coordinates[i])**2)),2) + h_values[i] for i in range(14)],start = 1))                                                               # Don't forget to change the value for key 9 in class node dictionary.
                f_values_2 = sorted(np.unique([i[1] for i in f_values]),reverse = True)
                present_node.f_dict = {i:[j[0] for j in f_values if j[1] == i and present_node.osf_dict[j[0]]!= 0] for i in f_values_2}
                # print(present_node.f_dict)

                value_list = []
                present_node.f_dict_2 = {}
                for i in present_node.f_dict:
                    value_list.extend(present_node.f_dict[i])
                while value_list != []:
                    k = value_list.pop()
                    present_node.f_dict_2[k] = [i for i in present_node.f_dict if k in present_node.f_dict[i]][0]

                #present_node.f_dict_2 = {j:i for i in present_node.f_dict for j in present_node.f_dict[i]}
                #print(present_node.f_dict)
                present_node.f_list = f_values_2
                #print(present_node.f_list)  
                #present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.f_list[-1]
                present_node.multidimensional_explore[stop_count]['f_list'] = copy.deepcopy(present_node.f_list)
                present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.f_list[-1] 
                present_node.multidimensional_explore[stop_count]['f_dict'] = copy.deepcopy(present_node.f_dict)
                                
            next_nodes = []                              ## Edit over here as I was restricting the motion to access further nodes by making it wait at a particular timestep.
            while next_nodes == []:                                     ## How 

                try:

                    change_in_f =  present_node.multidimensional_explore[stop_count]['f_list'].pop()   
                  
                except IndexError:
                    indexerror = True
                    break
                if dest_node.explored:
                    print('dest_node explored')
                
                for i in present_node.osf_dict:
                    if present_node.osf_dict[i] != 0 :
                        if stop_count + 1 not in present_node.osf_dict[i].multidimensional_explore:
                            present_node.osf_dict[i].multidimensional_explore[stop_count + 1] = copy.deepcopy(node_dict)
                            # print('successor = ' + str(present_node.osf_dict[i].multidimensional_explore[stop_count]['next']))
                next_nodes = [x for x in present_node.multidimensional_explore[stop_count]['f_dict'][change_in_f] if not (present_node.osf_dict[x].obstacle_node or present_node.osf_dict[x].multidimensional_explore[stop_count + 1]['explored'])]
                print(next_nodes)
                print(present_node.multidimensional_explore[stop_count]['f_dict'])  
                if again:
                    print('next_nodes = ' + str(next_nodes))
                           
            if indexerror:  
                indexerror = False
                continue
           
            # if present_node.multidimensional_explore[stop_count]['original_timestep'] != present_node.multidimensional_explore[stop_count]['timestep']:
            #     if [x for x in next_nodes if (agent_num, present_node.location, present_node.multidimensional_explore[stop_count]['timestep'] + 1) in present_node.osf_dict[x].waiting_constraints] != []:
            #         node_suppressed = True
            #         next_nodes = [x for x in next_nodes if (agent_num ,present_node.location, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].waiting_constraints]

            if [x for x in next_nodes if (agent_num , present_node.location , present_node.multidimensional_explore[stop_count]['timestep'] + 1) or (agent_num,present_node.multidimensional_explore[stop_count]['timestep'] + 1) in present_node.osf_dict[x].constraints] != []:
                node_suppressed = True
                next_nodes = [x for x in next_nodes if (agent_num ,present_node.location ,present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].constraints and (agent_num,present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].constraints]   ## Egge Constraints

            print(next_nodes)
            # if present_node.multidimensional_explore[stop_count]['original_timestep'] != 0:
            #     diff_in_time = present_node.multidimensional_explore[stop_count]['timestep'] - present_node.multidimensional_explore[stop_count]['original_timestep']
            #     if [x for x in next_nodes if (agent_num , present_node.multidimensional_explore[stop_count]['original_timestep'], present_node.multidimensional_explore[stop_count - diff_in_time]['predecessor'].location,x) in present_node.osf_dict[x].turning_constraints] != []:
            #         node_suppressed = True
            #         next_nodes = [x for x in next_nodes if (agent_num ,present_node.multidimensional_explore[stop_count]['original_timestep'], present_node.multidimensional_explore[stop_count - diff_in_time]['predecessor'].location,x) not in present_node.osf_dict[x].turning_constraints]
                
            if again:

                print('next_nodes again = ' + str(next_nodes))
                print(present_node.multidimensional_explore[stop_count]['f_dict'])
                time.sleep(2)
            
            if node_suppressed or change_in_f == present_node.f_list[0] or change_in_f >= 1:
                    if again:
                        print('reached here')
                        time.sleep(1)
                    if (agent_num, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.constraints and (not present_node.multidimensional_explore[stop_count]['wait']) :
                        new_node_creation = True
                            
            if new_node_creation:
                node_suppressed = False
                new_node_creation = False
            
                if (agent_num, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.constraints:
                    node_already_present = False
                    if stop_count + 1 in present_node.multidimensional_explore:
                        if present_node.multidimensional_explore[stop_count + 1]['explored']:
                            node_already_present = True
                            if again:
                                print('instead we had to jump here')
                                time.sleep(1)
                             
                    if not node_already_present:
                        just_created = True
                        if again:
                            print('we did reach here')

                            time.sleep(1)
                        present_node.multidimensional_explore[stop_count + 1] = copy.deepcopy(node_dict)
                      #  print('successor = ' + str(present_node.multidimensional_explore[stop_count + 1]['next']))
                        # if present_node.multidimensional_explore[stop_count]['original_timestep'] != 0 :
                        #     present_node.multidimensional_explore[stop_count + 1]['my_number'] = present_node.multidimensional_explore[stop_count]['my_number']

                        present_node.multidimensional_explore[stop_count + 1]['f_list'] = copy.deepcopy(present_node.f_list)
                        present_node.multidimensional_explore[stop_count + 1]['f_dict'] = copy.deepcopy(present_node.multidimensional_explore[stop_count]['f_dict'])
                    # new_node.dummy = True
                        #new_node.explored = True
                        present_node.multidimensional_explore[stop_count + 1]['explored'] = True
                        #new_node.timestep = present_node.timestep + 1
                        present_node.multidimensional_explore[stop_count + 1]['timestep'] = present_node.multidimensional_explore[stop_count]['timestep'] + 1
                        #new_node.f_static = present_node.f_static + 1
                        present_node.multidimensional_explore[stop_count + 1]['original_timestep'] =  present_node.multidimensional_explore[stop_count]['original_timestep']
                        present_node.multidimensional_explore[stop_count + 1]['f_static'] = present_node.multidimensional_explore[stop_count]['f_static'] + 1
                        #new_node.f_cost = change_in_f + present_node.f_static
                        present_node.multidimensional_explore[stop_count + 1]['f_cost'] = present_node.multidimensional_explore[stop_count + 1]['f_list'][-1] + present_node.multidimensional_explore[stop_count + 1]['f_static']
                        #new_node.predecessor = present_node
                        present_node.multidimensional_explore[stop_count + 1]['predecessor'] = present_node
                        present_node.multidimensional_explore[stop_count]['wait'] = True
                        low_level_nodes += 1
                        print('was here')
                        heapq.heappush(open,(present_node.multidimensional_explore[stop_count + 1]['f_static'],id(present_node),stop_count + 1,present_node))
                else:
                    if again:
                        print('we are here')
                        time.sleep(1)
            if (not just_created) and next_nodes == []:
                if len(open) != 0:
                    if present_node.multidimensional_explore[stop_count]['f_list'] !=[]:
                        no_node_added = True
                        continue
            
            just_created = False
            if present_node.multidimensional_explore[stop_count]['f_list'] != []:
                present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.multidimensional_explore[stop_count]['f_list'][-1]
                heapq.heappush(open,(present_node.multidimensional_explore[stop_count]['f_cost'],id(present_node),stop_count,present_node))
            
            if next_nodes != []:
 
                for i in next_nodes:
                    low_level_nodes += 1
                    #present_node.osf_dict[i].explored = True
                    # present_node.osf_dict[i].multidimensional_explore[stop_count]['my_number'] = i
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['explored'] = True
                    #present_node.osf_dict[i].predecessor = present_node
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['predecessor'] = present_node
                    #present_node.osf_dict[i].timestep = present_node.timestep + 1
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['timestep'] = present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['original_timestep'] = present_node.multidimensional_explore[stop_count]['timestep'] + 1
                    if present_node.osf_dict[i].f_list != []:
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_list'] = copy.deepcopy(present_node.osf_dict[i].f_list)
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_dict'] = copy.deepcopy(present_node.osf_dict[i].f_dict)
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_static'] = present_node.multidimensional_explore[stop_count]['f_static'] + change_in_f
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_cost'] = present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_static'] + present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['f_list'][-1]

                    heapq.heappush(open,(present_node.multidimensional_explore[stop_count]['f_static'] + change_in_f,id(present_node.osf_dict[i]),stop_count + 1,present_node.osf_dict[i]))
                    #print('locations of present_node and the suggested node = ' + str(present_node.location) + ' ' + str(present_node.osf_dict[i].location))
                    if present_node.osf_dict[i] == dest_node:
                        output = 1
                        break
            
            if output == 1:
                break
        
        if output == 1 :
            print('able to find')
            x_1,y_1 =  self.find_path_array_3D(final_node = dest_node,initial_node = start_node,stop_count = stop_count + 1) 
            # print('x_1 = ' + str(x_1))
            # print('y_1 = ' + str(y_1))
            # print('z_1 = ' + str(z_1))
            # print('d_1 = ' + str(d_1))
            
            return (x_1,y_1)        
        else:
            return None
    
    
    def find_path_3_D(self,agent_num,start_node = 1,dest_node = 2,timestep = 0, again = False):
        
        low_level_nodes = 0 
        set_of_nodes = copy.deepcopy(self.nodes)
        start_node = set_of_nodes[self.reference_dict[start_node.location]]
        dest_node = set_of_nodes[self.reference_dict[dest_node.location]]
        
        open = []
        a = np.max((np.abs(start_node.location[0] - dest_node.location[0]),np.abs(start_node.location[1] - dest_node.location[1])))
        heapq.heappush(open,(a,id(start_node),0,start_node)) 
        low_level_nodes += 1
        closed = []
        indexerror = node_suppressed = just_created = new_node_creation = False
        start_node.timestep = timestep 
        start_node.multidimensional_explore[0] = copy.deepcopy(node_dict)
        start_node.multidimensional_explore[0]['timestep'] = start_node.multidimensional_explore[0]['original_timestep'] = timestep
        
        output = 0
        no_node_added = False
        counter = 0
        while open != []:
            
            reached_here = False
            if no_node_added:
                no_node_added = False
                
            else:                                              ## Node ran out of neighbours due to constraints
                x,i_d,stop_count,present_node = heapq.heappop(open)
                counter += 1
                print(present_node.location)
                print(stop_count)
                    
            if present_node.f_list == [] :
                
                a,b,c = (abs(-present_node.location[0] + dest_node.location[0]), abs(- present_node.location[1] + dest_node.location[1]), abs(- present_node.location[2] + dest_node.location[2]))
                greater_value = max(a,b,c)
                # present_node.multidimensional_explore[stop_count]['f_static'] = x       # Chebychev distance
                # g = lambda x: 0 if x == 0 else (1 if x > 0 else -1)
                # direction = [g(a),g(b)]
                # if np.abs(a) == np.abs(b):
                #     h_values = osf_hexagon(direction,2)
                
                # else:
                #     c = np.abs(a) - np.abs(b)
                #     h_values = osf_hexagon(direction = direction,greater_direction = (lambda x : 0 if x > 0 else 1)(c), diff_one = True if abs(c) < 3/2 + square_root_three_half else False, direction_diff = abs(c))

                c_ = 1
                greatest_values_neighbours = np.array([(0, c_,0),(0,-c_,0),(c_,0,0),(-c_,0,0),(0,0,-c_),(0,0,c_)]) + present_node.location
                h_values_pre = np.array([max(tuple(i)) for i in list(np.abs(dest_node.location - greatest_values_neighbours))])
                h_values = list(h_values_pre - greater_value)
                f_values = list(enumerate([1 + h_values[i] for i in range(6)],start = 1))
                                                                        # Don't forget to change the value for key 9 in class node dictionary.
                f_values_2 = sorted(np.unique([i[1] for i in f_values]),reverse = True)
                present_node.f_dict = {i:[j[0] for j in f_values if j[1] == i and present_node.osf_dict[j[0]]!= 0] for i in f_values_2}
                # print(present_node.f_dict)

                value_list = []
                present_node.f_dict_2 = {}
                for i in present_node.f_dict:
                    value_list.extend(present_node.f_dict[i])
                while value_list != []:
                    k = value_list.pop()
                    present_node.f_dict_2[k] = [i for i in present_node.f_dict if k in present_node.f_dict[i]][0]

                #present_node.f_dict_2 = {j:i for i in present_node.f_dict for j in present_node.f_dict[i]}
                #print(present_node.f_dict)
                present_node.f_list = f_values_2
                #print(present_node.f_list)  
                #present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.f_list[-1]
                present_node.multidimensional_explore[stop_count]['f_list'] = copy.deepcopy(present_node.f_list)
                present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.f_list[-1] 
                present_node.multidimensional_explore[stop_count]['f_dict'] = copy.deepcopy(present_node.f_dict)
                                
            next_nodes = []                              ## Edit over here as I was restricting the motion to access further nodes by making it wait at a particular timestep.
            while next_nodes == []:                                     ## How 

                try:

                    change_in_f =  present_node.multidimensional_explore[stop_count]['f_list'].pop()   
                  
                except IndexError:
                    indexerror = True
                    break
                if dest_node.explored:
                    print('dest_node explored')
                
                for i in present_node.osf_dict:
                    if present_node.osf_dict[i] != 0 :
                        if stop_count + 1 not in present_node.osf_dict[i].multidimensional_explore:
                            present_node.osf_dict[i].multidimensional_explore[stop_count + 1] = copy.deepcopy(node_dict)
                            # print('successor = ' + str(present_node.osf_dict[i].multidimensional_explore[stop_count]['next']))
                next_nodes = [x for x in present_node.multidimensional_explore[stop_count]['f_dict'][change_in_f] if not (present_node.osf_dict[x].obstacle_node or present_node.osf_dict[x].multidimensional_explore[stop_count + 1]['explored'])]
                print(next_nodes)
                print(present_node.multidimensional_explore[stop_count]['f_dict'])  
                if again:
                    print('next_nodes = ' + str(next_nodes))
                           
            if indexerror:  
                indexerror = False
                continue
           
            # if present_node.multidimensional_explore[stop_count]['original_timestep'] != present_node.multidimensional_explore[stop_count]['timestep']:
            #     if [x for x in next_nodes if (agent_num, present_node.location, present_node.multidimensional_explore[stop_count]['timestep'] + 1) in present_node.osf_dict[x].waiting_constraints] != []:
            #         node_suppressed = True
            #         next_nodes = [x for x in next_nodes if (agent_num ,present_node.location, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].waiting_constraints]

            if [x for x in next_nodes if (agent_num , present_node.location , present_node.multidimensional_explore[stop_count]['timestep'] + 1) or (agent_num,present_node.multidimensional_explore[stop_count]['timestep'] + 1) in present_node.osf_dict[x].constraints] != []:
                node_suppressed = True
                next_nodes = [x for x in next_nodes if (agent_num ,present_node.location ,present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].constraints and (agent_num,present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.osf_dict[x].constraints]   ## Egge Constraints

            print(next_nodes)
            # if present_node.multidimensional_explore[stop_count]['original_timestep'] != 0:
            #     diff_in_time = present_node.multidimensional_explore[stop_count]['timestep'] - present_node.multidimensional_explore[stop_count]['original_timestep']
            #     if [x for x in next_nodes if (agent_num , present_node.multidimensional_explore[stop_count]['original_timestep'], present_node.multidimensional_explore[stop_count - diff_in_time]['predecessor'].location,x) in present_node.osf_dict[x].turning_constraints] != []:
            #         node_suppressed = True
            #         next_nodes = [x for x in next_nodes if (agent_num ,present_node.multidimensional_explore[stop_count]['original_timestep'], present_node.multidimensional_explore[stop_count - diff_in_time]['predecessor'].location,x) not in present_node.osf_dict[x].turning_constraints]
                
            if again:

                print('next_nodes again = ' + str(next_nodes))
                print(present_node.multidimensional_explore[stop_count]['f_dict'])
                time.sleep(2)
            
            if node_suppressed or change_in_f == present_node.f_list[0] or change_in_f >= 1:
                    if again:
                        print('reached here')
                        time.sleep(1)
                    if (agent_num, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.constraints and (not present_node.multidimensional_explore[stop_count]['wait']) :
                        new_node_creation = True
                            
            if new_node_creation:
                node_suppressed = False
                new_node_creation = False
            
                if (agent_num, present_node.multidimensional_explore[stop_count]['timestep'] + 1) not in present_node.constraints:
                    node_already_present = False
                    if stop_count + 1 in present_node.multidimensional_explore:
                        if present_node.multidimensional_explore[stop_count + 1]['explored']:
                            node_already_present = True
                            if again:
                                print('instead we had to jump here')
                                time.sleep(1)
                             
                    if not node_already_present:
                        just_created = True
                        if again:
                            print('we did reach here')

                            time.sleep(1)
                        present_node.multidimensional_explore[stop_count + 1] = copy.deepcopy(node_dict)
                      #  print('successor = ' + str(present_node.multidimensional_explore[stop_count + 1]['next']))
                        # if present_node.multidimensional_explore[stop_count]['original_timestep'] != 0 :
                        #     present_node.multidimensional_explore[stop_count + 1]['my_number'] = present_node.multidimensional_explore[stop_count]['my_number']

                        present_node.multidimensional_explore[stop_count + 1]['f_list'] = copy.deepcopy(present_node.f_list)
                        present_node.multidimensional_explore[stop_count + 1]['f_dict'] = copy.deepcopy(present_node.multidimensional_explore[stop_count]['f_dict'])
                    # new_node.dummy = True
                        #new_node.explored = True
                        present_node.multidimensional_explore[stop_count + 1]['explored'] = True
                        #new_node.timestep = present_node.timestep + 1
                        present_node.multidimensional_explore[stop_count + 1]['timestep'] = present_node.multidimensional_explore[stop_count]['timestep'] + 1
                        #new_node.f_static = present_node.f_static + 1
                        present_node.multidimensional_explore[stop_count + 1]['original_timestep'] =  present_node.multidimensional_explore[stop_count]['original_timestep']
                        present_node.multidimensional_explore[stop_count + 1]['f_static'] = present_node.multidimensional_explore[stop_count]['f_static'] + 1
                        #new_node.f_cost = change_in_f + present_node.f_static
                        present_node.multidimensional_explore[stop_count + 1]['f_cost'] = present_node.multidimensional_explore[stop_count + 1]['f_list'][-1] + present_node.multidimensional_explore[stop_count + 1]['f_static']
                        #new_node.predecessor = present_node
                        present_node.multidimensional_explore[stop_count + 1]['predecessor'] = present_node
                        present_node.multidimensional_explore[stop_count]['wait'] = True
                        low_level_nodes += 1
                        print('was here')
                        heapq.heappush(open,(present_node.multidimensional_explore[stop_count + 1]['f_static'],id(present_node),stop_count + 1,present_node))
                else:
                    if again:
                        print('we are here')
                        time.sleep(1)
            if (not just_created) and next_nodes == []:
                if len(open) != 0:
                    if present_node.multidimensional_explore[stop_count]['f_list'] !=[]:
                        no_node_added = True
                        continue
            
            just_created = False
            if present_node.multidimensional_explore[stop_count]['f_list'] != []:
                present_node.multidimensional_explore[stop_count]['f_cost'] = present_node.multidimensional_explore[stop_count]['f_static'] + present_node.multidimensional_explore[stop_count]['f_list'][-1]
                heapq.heappush(open,(present_node.multidimensional_explore[stop_count]['f_cost'],id(present_node),stop_count,present_node))
            
            if next_nodes != []:
 
                for i in next_nodes:
                    low_level_nodes += 1
                    #present_node.osf_dict[i].explored = True
                    # present_node.osf_dict[i].multidimensional_explore[stop_count]['my_number'] = i
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['explored'] = True
                    #present_node.osf_dict[i].predecessor = present_node
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['predecessor'] = present_node
                    #present_node.osf_dict[i].timestep = present_node.timestep + 1
                    present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['timestep'] = present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['original_timestep'] = present_node.multidimensional_explore[stop_count]['timestep'] + 1
                    if present_node.osf_dict[i].f_list != []:
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_list'] = copy.deepcopy(present_node.osf_dict[i].f_list)
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_dict'] = copy.deepcopy(present_node.osf_dict[i].f_dict)
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_static'] = present_node.multidimensional_explore[stop_count]['f_static'] + change_in_f
                        present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_cost'] = present_node.osf_dict[i].multidimensional_explore[stop_count+1]['f_static'] + present_node.osf_dict[i].multidimensional_explore[stop_count + 1]['f_list'][-1]

                    heapq.heappush(open,(present_node.multidimensional_explore[stop_count]['f_static'] + change_in_f,id(present_node.osf_dict[i]),stop_count + 1,present_node.osf_dict[i]))
                    #print('locations of present_node and the suggested node = ' + str(present_node.location) + ' ' + str(present_node.osf_dict[i].location))
                    if present_node.osf_dict[i] == dest_node:
                        output = 1
                        break
            
            if output == 1:
                break
        
        if output == 1 :
            
            x_1,y_1 =  self.find_path_array_3D(final_node = dest_node,initial_node = start_node,stop_count = stop_count + 1) 
            # print('x_1 = ' + str(x_1))
            # print('y_1 = ' + str(y_1))
            # print('z_1 = ' + str(z_1))
            # print('d_1 = ' + str(d_1))
            
            return (x_1,y_1)        
        else:
            return None


    def check_conflicts(self,diagonal = False):
        
        already_checked = False
        initial_number = 0
        for i in range(len(self.node_grid)):
            i_1 = self.node_grid[i:i+2]
            k = 0            
            k_2 = 0
            
            if len(i_1) == 1:
                the_number = 1
            else:
                the_number = 2
            for m in range(initial_number,the_number):
                list_to_remember = []
                if i + m != 0:  
                    
                    for j in i_1[m]:
                        j.explored = False
                        j.agent_list = []

                    k_2 = 0
                    for j in i_1[m]:
                        
                        k_2 += 1
                        if not j.explored:
                            
                            if self.node_grid[-1][k_2 - 1] == j and self.node_grid[i + m - 1][k_2 - 1] == j:
                                continue
                            
                            j.explored = True
                            j.agent_list.append(k_2)
                            #the_imaginary_agent_list = [(j.agent_list[-1],p) for p in j.imaginary_agent_list]  
                            if self.node_grid[i + m - 1][k_2 - 1] != j:
                            
                                if j.imaginary_agent_list != []:
                                    list_of_prev_locations = []
                                    for p in j.imaginary_agent_list:
                                        list_of_prev_locations.append((p,self.node_grid[i + m - 1][p-1]))
                                    list_of_prev_locations_2 = []
                                    
                                    for e in list_of_prev_locations:
                                        e_ = [x for x in range(1,7) if j.osf_dict[x] == e[1]][0]
                                        list_of_prev_locations_2.append((e[0],e_))

                                    list_of_prev_locations = list_of_prev_locations_2
                                    number_of_the_real_move = [x for x in range(1,7) if j.osf_dict[x] == self.node_grid[i + m - 1][k_2 - 1]][0]
                                    list_of_prev_locations = [e for e in list_of_prev_locations if e[1] in [generate_neighbours(number_of_the_real_move + 1),generate_neighbours(number_of_the_real_move + 2)]]
                                    try:
                                        agent_in_conflict = list_of_prev_locations.pop()[0]
                                    except IndexError:
                                        continue
 
                                    required_dict = {'imaginary_agent': agent_in_conflict,'real_agent': k_2, 'timestep': i + m}    
                                    for _ in i_1[m]:
                                        _.explored = False
                                        _.agent_list = []

                                    for _ in list_to_remember:
                                        _.imaginary_agent_list = []         

                                    return required_dict
                        else:

                            if self.node_grid[-1][k_2 - 1] == j and self.node_grid[i + m - 1][k_2 - 1] == j:
                                continue
                            #print('Agent_list for j = ' + str(j.agent_list))
                            j.agent_list.append(k_2)
                            required_tuple = (j,j.agent_list,i + m)

                            for _ in i_1[m]:
                                _.explored = False
                                _.agent_list = []
                            for _ in list_to_remember:
                                _.imaginary_agent_list = []
                            
                            return required_tuple
                    for _ in i_1[m]:
                        
                        _.explored = False
                        _.agent_list = []
                    
                    for _ in list_to_remember:
                       
                        _.imaginary_agent_list = []
            k += 1
            list_to_remember = []
            try : 
                f_ = list(zip(i_1[0],i_1[1]))
            except IndexError:
                return None

            for i_5,i_6 in f_:
                i_5.successor = []
                i_6.successor = []
                i_6.predecessor = []
                i_5.predecessor = []
                i_5.agent_list = []
                i_6.agent_list = [] 
    
            imaginary_dict = {}
            imaginary_dict_2 = {}
            
            k_2 = 0
           
            for i_3 in f_:
                
                k_2 += 1
                a,b = i_3
              
                if a == b:
                    continue
              
             
                if b in a.predecessor :

                    required_tuple = ((a,b),[k_2,imaginary_dict[(b,a)]],i + k)
                    for i_5,i_6 in imaginary_dict:
                        i_5.successor = i_6.successor = i_6.predecessor =  i_5.predecessor = []
                    
                    return required_tuple
                
                if diagonal:
                    if (round((a.location[0]+ b.location[0])/2,2),round((a.location[1]+ b.location[1])/2,2),round((a.location[2]+ b.location[2])/2,2)) in imaginary_dict_2 :
                        k_3 = imaginary_dict_2[(round((a.location[0]+ b.location[0])/2,2),round((a.location[1]+ b.location[1])/2,2),round((a.location[2]+ b.location[2])/2,2))]
                        required_tuple = (((a,b),k_2,i + k),(f_[k_3-1],k_3,i+k))
                        for i_5,i_6 in imaginary_dict:
                            i_5.successor = i_6.successor = i_6.predecessor =  i_5.predecessor = []
                        
                        return required_tuple
                
                a.successor.append(b) 
                b.predecessor.append(a)
                imaginary_dict[(a,b)] = k_2   
                if diagonal:
                    imaginary_dict_2[(round((a.location[0] + b.location[0])/2,2),round((a.location[1]+ b.location[1])/2,2),round((a.location[2]+ b.location[2])/2,2))] = k_2
    
        return None
    
    def check_conflicts_variant(self,the_independent_list = []):
        
        
        already_checked = False
        initial_number = 0
        for i in range(len(self.node_grid)):
            i_1 = self.node_grid[i:i+2]
            k = 0            
            k_2 = 0
            
            if len(i_1) == 1:
                the_number = 1
            else:
                the_number = 2
            for m in range(initial_number,the_number):
                list_to_remember = []
                if i + m != 0:  
                    
                    if i + m != 1:
                        
                        the_wait_dict = {}
                        for _ in range(1,len(self.agents)):

                            if i + m - 1 in self.future_positions:
                                if _ in self.future_positions[i + m - 1]:

                                    if self.future_positions[i + m - 1][_] != 0:

                                        list_to_remember.append(self.reference_nodes[self.reference_dict[self.future_positions[i + m - 1][_]]])
                                        if list_to_remember[-1].imaginary_agent_list != []:
                                            list_to_remember[-1].imaginary_agent_list.append(_)
                                            the_wait_dict[_] = [e for e in self.node_grid[i + m - 1][_ - 1].osf_dict if self.node_grid[i + m - 1][_ - 1].osf_dict[e] == list_to_remember[-1]][0] 
                                            
                                            the_new_list = [(e,the_wait_dict[e]) for e in list_to_remember[-1].imaginary_agent_list if e != _]
                                            the_new_list = [e for e in the_new_list if e[1] != generate_neighbours(the_wait_dict[_] + 3)]

                                            if the_new_list != []:
                                              
                                                if the_independent_list != []:
                                                    while the_new_list != []:
                                                        the_new_tuple = the_new_list.pop()
                                                        for u in the_independent_list:
                                                            if the_new_tuple[0] in u and _ in u:
                                                                agent_dict = {_:[[e for e in self.node_grid[i + m - 1][_ - 1].osf_dict if self.node_grid[i + m - 1][_ - 1].osf_dict[e] == self.node_grid[i + m - 2][_ - 1]][0], the_wait_dict[_],'wait'],the_new_tuple[0]:[],'wait': True, 'timestep': i + m - 1}
                                                                agent_dict[the_new_tuple[0]] = [[e for e in self.node_grid[i + m - 1][the_new_tuple[0] - 1].osf_dict if self.node_grid[i + m - 1][the_new_tuple[0] - 1].osf_dict[e] == self.node_grid[i + m - 2][the_new_tuple[0] - 1]][0],the_wait_dict[the_new_tuple[0]],'wait']
                                                                for e in list_to_remember:
                                                                    e.imaginary_agent_list = []
                                                                return agent_dict
                                                else:

                                                    for e in list_to_remember:
                                                        e.imaginary_agent_list = []
                                                    
                                                    the_new_tuple = the_new_list[0]
                                                    agent_dict = {_:[[e for e in self.node_grid[i + m - 1][_ - 1].osf_dict if self.node_grid[i + m - 1][_ - 1].osf_dict[e] == self.node_grid[i + m - 2][_ - 1]][0], the_wait_dict[_],'wait'],the_new_tuple[0]:[],'wait': True, 'timestep': i + m - 1}
                                                    agent_dict[the_new_tuple[0]] = [[e for e in self.node_grid[i + m - 1][the_new_tuple[0] - 1].osf_dict if self.node_grid[i + m - 1][the_new_tuple[0] - 1].osf_dict[e] == self.node_grid[i + m - 2][the_new_tuple[0] - 1]][0],the_wait_dict[the_new_tuple[0]],'wait']
                                                    return agent_dict
                                
                                        else:
                                            list_to_remember[-1].imaginary_agent_list.append(_)
                                            the_wait_dict[_] = [e for e in self.node_grid[i + m - 1][_ - 1].osf_dict if self.node_grid[i + m - 1][_ - 1].osf_dict[e] == list_to_remember[-1]][0]
        
                    for _ in range(1,len(self.agents)):
                         
                        try:
                            the_list = self.agent_positions[i + m][_]
                             
                        except KeyError:
                            continue
                       
                        if the_list != []:
                            # print('the_list = ' + str(the_list))
                            # print(self.node_grid[i + m - 2][_ - 1].location)
                            # print(self.node_grid[i + m - 1][_ - 1].location)
                            # print([self.node_grid[i + m - 2][_ - 1].osf_dict[e].location for e in range(1,7) if self.node_grid[i + m - 2][_ - 1].osf_dict[e] != 0 ])
                            the_key = [e for e in range(1,7) if self.node_grid[i + m - 2][_ - 1].osf_dict[e] == self.node_grid[i + m - 1][_ - 1]][0]
                        
                        for __ in the_list:
                           
                            the_next_key = [e for e in range(1,7) if self.node_grid[i + m - 1][_ - 1].osf_dict[e] == self.reference_nodes[self.reference_dict[__]]][0]
                            if the_next_key != the_key:
                                continue
                            
                            list_to_remember.append(self.reference_nodes[self.reference_dict[__]])
                            list_to_remember[-1].imaginary_agent_list.append(_)

                    list_to_remember = list(set(list_to_remember))
                    for j in i_1[m]:
                        j.explored = False
                        j.agent_list = []

                    k_2 = 0
                    for j in i_1[m]:
                        
                        k_2 += 1
                        if not j.explored:
                            
                            if self.node_grid[-1][k_2 - 1] == j and self.node_grid[i + m - 1][k_2 - 1] == j:
                                continue
                            
                            j.explored = True
                            j.agent_list.append(k_2)
                            #the_imaginary_agent_list = [(j.agent_list[-1],p) for p in j.imaginary_agent_list]  
                            if self.node_grid[i + m - 1][k_2 - 1] != j:
                            
                                if j.imaginary_agent_list != []:
                                    list_of_prev_locations = []
                                    for p in j.imaginary_agent_list:
                                        list_of_prev_locations.append((p,self.node_grid[i + m - 1][p-1]))
                                    list_of_prev_locations_2 = []
                                    
                                    for e in list_of_prev_locations:
                                        e_ = [x for x in range(1,7) if j.osf_dict[x] == e[1]][0]
                                        list_of_prev_locations_2.append((e[0],e_))

                                    list_of_prev_locations = list_of_prev_locations_2
                                    number_of_the_real_move = [x for x in range(1,7) if j.osf_dict[x] == self.node_grid[i + m - 1][k_2 - 1]][0]
                                    list_of_prev_locations = [e for e in list_of_prev_locations if e[1] in [generate_neighbours(number_of_the_real_move + 1),generate_neighbours(number_of_the_real_move + 2)]]
                                    try:
                                        agent_in_conflict = list_of_prev_locations.pop()[0]
                                    except IndexError:
                                        continue
 
                                    required_dict = {'imaginary_agent': agent_in_conflict,'real_agent': k_2, 'timestep': i + m}    
                                    if the_independent_list != []:
                                        for u in the_independent_list:
                                            if required_dict['imaginary_agent'] in u and required_dict['real_agent'] in u:

                                                for _ in i_1[m]:
                                                    _.explored = False
                                                    _.agent_list = []

                                                for _ in list_to_remember:
                                                    _.imaginary_agent_list = []         

                                                return required_dict
                                            
                                    else:

                                        for _ in i_1[m]:
                                            _.explored = False
                                            _.agent_list = []

                                        for _ in list_to_remember:
                                            _.imaginary_agent_list = []         

                                        return required_dict
                        else:

                            if self.node_grid[-1][k_2 - 1] == j and self.node_grid[i + m - 1][k_2 - 1] == j:
                                # print('this constraint wasnt required at all')
                                # print('location = ' + str(j.location))
                                # print(k_2)
                                continue
                            #print('Agent_list for j = ' + str(j.agent_list))
                            j.agent_list.append(k_2)
                            required_tuple = (j,j.agent_list,i + m)
                            if the_independent_list != []:
                                for u in the_independent_list:
                                    if j.agent_list[0] in u and  j.agent_list[1] in u:
                            
                                        for _ in i_1[m]:
                                            _.explored = False
                                            _.agent_list = []
                                        for _ in list_to_remember:
                                            _.imaginary_agent_list = []
                                        
                                        return required_tuple
                                    
                            else:

                                for _ in i_1[m]:
                                    _.explored = False
                                    _.agent_list = []
                                for _ in list_to_remember:
                                    _.imaginary_agent_list = []
                                
                                return required_tuple
                    
                    for _ in i_1[m]:
                        
                        _.explored = False
                        _.agent_list = []
                    
                    for _ in list_to_remember:
                       
                        _.imaginary_agent_list = []
            k += 1
            list_to_remember = []
            try : 
                f_ = list(zip(i_1[0],i_1[1]))
            except IndexError:
                return None

            for i_5,i_6 in f_:
                i_5.successor = []
                i_6.successor = []
                i_6.predecessor = []
                i_5.predecessor = []
                i_5.agent_list = []
                i_6.agent_list = [] 
    
            imaginary_dict = {}
            
            k_2 = 0
            # print('the start')
            for i_3 in f_:
                
                k_2 += 1
                a,b = i_3
                # print(a.location,b.location)
                if a == b:
                    continue

                
                # print('in the middle')
                
                # print(a.successor)
                # print(a.predecessor)
                # print(b.successor)
                # print(b.predecessor)

              #  print('in the end')
                if b in a.predecessor :
                   
                    if the_independent_list != []:
                        for u in the_independent_list:
                            if a in u or b in u:
 
                                required_tuple = ((a,b),[k_2,imaginary_dict[(b,a)]],i + k)
                                
                                for i_5,i_6 in imaginary_dict:
                                    i_5.successor = i_6.successor = i_6.predecessor =  i_5.predecessor = []
                                    # i_5.agent_list = i_6.agent_list = []     
                                
                                return required_tuple
                            
                    else:

                        required_tuple = ((a,b),[k_2,imaginary_dict[(b,a)]],i + k)
                                
                        for i_5,i_6 in imaginary_dict:
                            i_5.successor = i_6.successor = i_6.predecessor =  i_5.predecessor = []
                            # i_5.agent_list = i_6.agent_list = []     
                        
                        return required_tuple
            
                # print('after the end')
                a.successor.append(b) 
                b.predecessor.append(a)
                # print(a.successor)
                # print(b.predecessor)
                # print('the very end')
                imaginary_dict[(a,b)] = k_2
           # print('the end')   
            
          #  print('over here')
            # for a,b in f_:
                
            #     # print(a.successor)
            #     # print(a.predecessor)
            #     # print(b.successor)
            #     # print(b.predecessor)

            for _ in range(1,len(self.agents)):
                    
                    
                    try:
                        
                        the_list = self.agent_positions[i + k][_]
                        if _ == 8:
                            print('For eight = '+ str([the_list]))
                        try:
                            
                            the_future_location = self.future_positions[i + k - 1][_]
                            if the_future_location != 0:
                                
                                the_future_index = [e for e in range(1,7) if self.node_grid[i + k - 1][_ - 1].osf_dict[e] == self.reference_nodes[self.reference_dict[the_future_location]]][0]
                                
                                if self.node_grid[i - 1][_ - 1].osf_dict[the_future_index] != self.node_grid[i + k - 1][_ - 1]:  ## Over here we are just getting rid of situations where the vehicle takes a turn before waiting.
                                    the_list = [e for e in the_list if e != the_future_location]
                        
                        except KeyError:
                            pass
                    
                    except KeyError:
                        continue
                    
                    for __ in the_list:
                       
                        imaginary_dict[(self.node_grid[i][_ - 1],self.reference_nodes[self.reference_dict[__]])] = _
                        if self.reference_nodes[self.reference_dict[__]] in self.node_grid[i][_ - 1].predecessor:
                            # print('the agent = ' + str(_))
                            # print('the list = ' + str(the_list))
                            # print([(a.location,b.location) for a,b in imaginary_dict])
                            # print((self.node_grid[i][_ - 1].location,self.reference_nodes[self.reference_dict[__]].location))
                            # print([e.location for e in self.reference_nodes[self.reference_dict[__]].successor])
                            # print([e.location for e in self.node_grid[i][_ - 1].predecessor])
                            the_other_agent = imaginary_dict[(self.reference_nodes[self.reference_dict[__]],self.node_grid[i][_ - 1])]
                            
                            agent_dict = {_ : [],the_other_agent : [],'wait':False}
                            found_conflict = False
                            if the_independent_list != []:
                                for u in the_independent_list:
                                    if _ in u and the_other_agent in u:
                                        found_conflict = True
                                        break
                            # print(agent_dict)
                            # print('timestep = ' + str(i))
                            # print(imaginary_dict)
                            if the_independent_list != []:
                                if not found_conflict:
                                    continue

                            for ___ in list(agent_dict.keys())[:2]:
                                # print(self.node_grid[i][___ - 1].osf_dict)
                                try:
                                    agent_dict[___].append([e for e in range(1,7) if self.node_grid[i][___ - 1].osf_dict[e] == self.node_grid[i - 1][___ - 1]][0])
                                except IndexError:
                                    agent_dict['waiting_agent'] = ___
                                if self.node_grid[i][___ - 1] != self.node_grid[i + 1][___ - 1]:
                                    agent_dict[___].append([e for e in range(1,7) if self.node_grid[i][___ - 1].osf_dict[e] == self.node_grid[i + 1][___ - 1]][0])
                                else:
                                    agent_dict[___].append([e for e in range(1,7) if self.node_grid[i][___ - 1].osf_dict[e] == self.reference_nodes[self.reference_dict[self.future_positions[i][___]]]][0])
                                    agent_dict[___].append(True)
                                    agent_dict['wait'] = True
                                
                                if len(agent_dict[___]) != 1:
                                    agent_dict[___][0] = generate_neighbours(agent_dict[___][0] + 3)

                            if 'waiting_agent' in agent_dict:

                                for i_5,i_6 in imaginary_dict:
                                        
                                    i_5.successor = []
                                    i_6.successor = []
                                    i_6.predecessor = []
                                    i_5.predecessor = []

                                agent_dict['timestep'] = i

                                return agent_dict
  
                            the_numbers = { ___ : number_assignments(np.abs(agent_dict[___][0] - agent_dict[___][1])) for ___ in list(agent_dict.keys())[:2]}
                            the_numbers_list = [the_numbers[___] for ___ in the_numbers]
                            
                            if 2 in the_numbers_list:

                                if agent_dict['wait'] or 1 in the_numbers_list:
                                    continue
                                
                                else:

                                    for i_5,i_6 in imaginary_dict:
                                        
                                        i_5.successor = []
                                        i_6.successor = []
                                        i_6.predecessor = []
                                        i_5.predecessor = []
                                   
                                    agent_dict['timestep'] = i

                                    return agent_dict

                            elif the_numbers_list == [1,1]:

                                if agent_dict[_][0] != generate_neighbours(agent_dict[the_other_agent][0] + 3):
                                    continue

                                
                                if True not in [True if generate_neighbours(agent_dict[___][0] + 1) == agent_dict[___][1] else False for ___ in list(agent_dict.keys())[:2]]:
                                    continue
                                
                                else:
                                    
                                    for i_5,i_6 in imaginary_dict:
                                        
                                        i_5.successor = []
                                        i_6.successor = []
                                        i_6.predecessor = []
                                        i_5.predecessor = []
                                    
                                    agent_dict['timestep'] = i
                                    return agent_dict   

                            else:
                                    for i_5,i_6 in imaginary_dict:
                                        
                                        i_5.successor = []
                                        i_6.successor = []
                                        i_6.predecessor = []
                                        i_5.predecessor = []
                                   
                                    agent_dict['timestep'] = i
                                    return agent_dict

                        else:
                            self.node_grid[i][_ - 1].successor.append(self.reference_nodes[self.reference_dict[__]])
                            self.reference_nodes[self.reference_dict[__]].predecessor.append(self.node_grid[i][_ - 1])
          
            for i_5,i_6 in imaginary_dict:

                i_5.successor = []
                i_6.successor = []
                i_6.predecessor = [] 
                i_5.predecessor = []
              #  i_5.agent_list = i_6.agent_list = [] 
            
            already_checked = True
            initial_number = 1
        return None
        
    def arrange_node_grid(self,returned_path = [],agent_num:int = 0,exceeds = False):
        
        # if agent_num == 16:
        #     print('full returned_path = ' + str([x.location for x in returned_path]))
        
        length = self.node_grid.shape[0]
        if self.timestep >= self.timestep_start_dict[agent_num][0]:
            returned_path = returned_path[self.timestep - self.timestep_start_dict[agent_num][0]:]

        else :
            returned_path = list(self.node_grid.T[agent_num-1])[:self.timestep_start_dict[agent_num][0] - self.timestep] + returned_path
        
        if False in [True if x == 0 or (returned_path[x] in returned_path[x-1].osf_dict.values() or returned_path[x] == returned_path[x-1]) else False for x in range(len(returned_path))]:
            print('Flawed_node_list = ' + str([x.location for x in self.node_grid.T[agent_num-1]]))
            return 1
        #print('Cut returned path = ' + str([x.location for x in returned_path]))
        #print('timestep = ' + str(self.timestep))
       #print('timestep_start_dict = ' + str(self.timestep_start_dict[agent_num][0]))

        # if agent_num == 16:
        #     print('returned_path = ' + str([x.location for x in returned_path]))
        #     print('timestep = ' + str(self.timestep))
        if len(returned_path) > length:
            exceeds = True
        
        if not exceeds:
            self.node_grid.T[agent_num-1][:len(returned_path)] = returned_path
            if len(returned_path) < length:
                self.node_grid.T[agent_num-1][len(returned_path):] = [returned_path[-1] for i in range(length - len(returned_path))]
            if False in [True if x == 0 or (self.node_grid.T[agent_num-1][x] in self.node_grid.T[agent_num-1][x-1].osf_dict.values() or self.node_grid.T[agent_num-1][x] == self.node_grid.T[agent_num-1][x-1]) else False for x in range(len(self.node_grid.T[agent_num-1]))]:
                print('Flawed_node_list = ' + str([x.location for x in self.node_grid.T[agent_num-1]]))
                return 1
        else:
            node_grid = []
            for i in range(len(self.node_grid.T)):
                node_grid.append(np.pad(self.node_grid.T[i],(0,len(returned_path) - length),mode = 'constant',constant_values = (0,self.node_grid.T[i][-1])))
            
            self.node_grid = (np.array(node_grid)).T
            self.node_grid.T[agent_num-1][:len(returned_path)] = returned_path
            if False in [True if x == 0 or (self.node_grid.T[agent_num-1][x] in self.node_grid.T[agent_num-1][x-1].osf_dict.values() or self.node_grid.T[agent_num-1][x] == self.node_grid.T[agent_num-1][x-1]) else False for x in range(len(self.node_grid.T[agent_num-1]))]:
                print('Flawed_node_list = ' + str([x.location for x in self.node_grid.T[agent_num-1]]))
                return 1
            return None
        
        return None
 