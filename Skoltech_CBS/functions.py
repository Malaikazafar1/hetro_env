import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from classes_correction import *

a = 3 / 2
b = np.round((np.sqrt(3)) / 2, 2)
c = np.round(np.sqrt(3), 2)
dir_tuple_2 = [(0, c), (-a, b), (-a, -b), (0, -c), (a, -b), (a, b)]
dir_tuple = [(0, 1), (-1, 1), (-1, -1), (0, -1), (1, -1), (1, 1)]
change_in_dir = list(zip(*dir_tuple))
change_in_dir = [list(i) for i in change_in_dir]
change_in_dir_2 = list(zip(*dir_tuple_2))
change_in_dir_2 = [list(i) for i in change_in_dir_2]



def generate_neighbours(x):
    if x == 6 or x == 0:
        return 6
    else:
        return x % 6


def osf(direction, greater_direction, diff_one=False):
    dir_tuple = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    change_in_dir = list(zip(*dir_tuple))
    if not diff_one:

        if greater_direction != 2:
            return (
                np.array(list(change_in_dir[greater_direction]))
                * direction[greater_direction]
                * (-1)
            )

        else:
            return [
                np.min((a * direction[0], b * direction[1])) * (-1)
                for a, b in dir_tuple
            ]

    else:
        g = lambda x: 1 if x == 0 else 0
        f = lambda x: 2 if x == -1 else 0
        return [
            (
                -i[greater_direction] * direction[greater_direction]
                if i[greater_direction] * direction[greater_direction] != 1
                else -1 + f(i[g(greater_direction)] * direction[g(greater_direction)])
            )
            for i in dir_tuple
        ]


def osf_hexagon(
    direction, greater_direction, diff_one=0, direction_tuple=(), direction_diff=0
):

    if not diff_one:

        if greater_direction != 2:
            # print('reached here where ty')
            return (
                np.array(list(change_in_dir_2[greater_direction]))
                * direction[greater_direction]
                * (-1)
            )

        else:
            # print('reached herepe != 2')
            return [
                np.min((a * direction[0], b * direction[1])) * (-1)
                for a, b in dir_tuple_2
            ]

    else:

        f_0 = [
            i * direction[greater_direction - 1] * (-1)
            for i in change_in_dir_2[greater_direction - 1]
        ]
        f = [
            i * (-1) * direction[greater_direction]
            for i in change_in_dir[greater_direction]
        ]
        f_2 = [
            i * (-1) * direction[greater_direction - 1]
            for i in change_in_dir[greater_direction - 1]
        ]
        f_3 = list(map(lambda a, b: a * b, f, f_2))
        h_list = [
            i * direction[greater_direction] * (-1)
            for i in change_in_dir_2[greater_direction]
        ]
        if diff_one == 1:
            print("reached here where diff == 1")
            return [
                (
                    h_list[i]
                    if f_3[i] != -1
                    else (h_list[i] if f[i] != -1 else -direction_diff + f_0[i])
                )
                for i in range(6)
            ]
        else:
            print("reached here where diff == 2")
            return [
                (
                    h_list[i]
                    if f_3[i] == 1
                    else (
                        h_list[i]
                        if f[i] != -1 and not (f[i] == 0 and f_2[i] == 1)
                        else -direction_diff + f_0[i]
                    )
                )
                for i in range(6)
            ]


def find_path(dest_node, initial_node):

    node_list = []
    x = dest_node
    node_list.append(x)
    while x != initial_node:
        x = x.predecessor
        if x.dummy:

            t = 0
            while x.dummy:
                t += 1
                x = x.predecessor

            node_list.extend([x for i in range(t)])
        node_list.append(x)

    time = dest_node.timestep
    node_list.reverse()
    return (node_list, time)


def in_middle_3_D(coordinate, dimension_extremes):

    if (
        coordinate[0] - dimension_extremes[0][0] <= 0
        or dimension_extremes[0][1] - coordinate[0] <= 0
       or coordinate[1] - dimension_extremes[1][0] <= 0
        or dimension_extremes[1][1] - coordinate[1] <= 0
        or coordinate[2] - dimension_extremes[2][0] <= 0
        or dimension_extremes[2][1] - coordinate[2] <= 0
         
    ):
        return False

    return True

def in_middle(coordinate, dimension_extremes):

    if (
        coordinate[0] - dimension_extremes[0][0] <= 0
        or dimension_extremes[0][1] - coordinate[0] <= 0
        or dimension_extremes[1][1] - coordinate[1] <= 0
        or coordinate[1] - dimension_extremes[1][0] <= 0
    ):
        return False

    return True

def generate_grid(
    initial_location,
    dimension_extremes,
    direction_list=[
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    location_list=[],
    node_list=[],
):

    print(in_middle(initial_location, dimension_extremes))
    if (
        not in_middle(initial_location, dimension_extremes)
    ) or initial_location in location_list:
        return None
    print(1)
    x = low_level_node(location=initial_location)
    location_list.append(initial_location)

    node_list.append(x)
    k = 0

    for i in np.array(direction_list) + np.array([initial_location]):
        k += 1

        m = generate_grid(
            tuple(i),
            dimension_extremes,
            direction_list,
            location_list=location_list,
            node_list=node_list,
        )

        if m == None:
            continue

        else:
            x.osf_dict[k], node_list, location_list = m

    return (x, node_list, location_list)


a = 3 / 2
b = (math.sqrt(3)) / 2
c = math.sqrt(3)


def generate_grid_hexagon(
    initial_location,
    dimension_extremes,
    direction_list=[(0, c), (-a, b), (-a, -b), (0, -c), (a, -b), (a, b)],
    location_list=[],
    node_list=[],
):

    if (
        (not in_middle(initial_location, dimension_extremes))
        or initial_location in location_list
        or [
            (x, y)
            for x, y in location_list
            if (x - initial_location[0]) ** 2 + (y - initial_location[1]) ** 2 <= 1
        ]
        != []
    ):
        return None
    if not in_middle(initial_location, dimension_extremes):
        print(initial_location)
    x = low_level_node_hexagon(location=initial_location)
    location_list.append(initial_location)

    node_list.append(x)
    print(len(node_list))
    k = 0

    for i in np.array(direction_list) + np.array([initial_location]):
        x_1, y_1 = tuple(i)
        k += 1

        m = generate_grid_hexagon(
            (x_1, y_1),
            dimension_extremes,
            direction_list,
            location_list=location_list,
            node_list=node_list,
        )

        if m == None:
            continue

        else:
            x.osf_dict[k], node_list, location_list = m
    return (x, node_list, location_list)

a_= 1
b_ = 1
c_ = 1

def generate_grid_hexagon_3D(
    initial_location,
    dimension_extremes,
    direction_list=[(0, c_,0),(0,-c_,0),(c_,0,0),(-c_,0,0),(0,0,-c_),(0,0,c_)],
    location_list=[],
    node_list=[],
):

    if (
        (not in_middle_3_D(initial_location, dimension_extremes))
        or initial_location in location_list
        or [
            (x, y, z)
            for x, y, z in location_list
            if (x - initial_location[0]) ** 2 + (y - initial_location[1]) ** 2 + (z - initial_location[2]) ** 2 < 1      
        ]
        != []
    ):
        print("None")
        return None
    if not in_middle_3_D(initial_location, dimension_extremes):
        print(initial_location)
    x = low_level_node_hexagon(location=initial_location)
    location_list.append(initial_location)

    node_list.append(x)
    print(len(node_list))
    k = 0

    for i in np.array(direction_list) + np.array([initial_location]):
        print('i = ' + str(i))
        x_1, y_1, z_1 = tuple(i)
        k += 1

        m = generate_grid_hexagon_3D(
            (x_1, y_1,z_1),
            dimension_extremes,
            direction_list,
            location_list=location_list,
            node_list=node_list,
        )

        if m == None:
            continue

        else:
            x.osf_dict[k], node_list, location_list = m
    return (x, node_list, location_list)

def check_conflicts(x):

    for i in range(len(x)):
        i_1 = x[i : i + 2]
        k = 0
        k_2 = 0
        for j in i_1[k]:
            j.explored = False
            j.agent_list = []

        for j in i_1[k]:

            k_2 += 1
            if not j.explored:

                j.explored = True
                j.agent_list.append(k_2)
            else:
                print("Agent_list for j = " + str(j.agent_list))
                j.agent_list.append(k_2)
                required_tuple = (j, j.agent_list, i + k)
                print(j.agent_list)
                print("Executed before vertex conflict")
                for _ in i_1[k]:
                    _.explored = False
                    _.agent_list = []

                return required_tuple
        for _ in i_1[k]:
            _.explored = False
            _.agent_list = []
        k += 1
        try:
            _ = list(zip(i_1[0], i_1[1]))
        except IndexError:
            return None

        for i_5, i_6 in _:
            i_5.successor = i_6.successor = i_6.predecessor = i_5.predecessor = 0
            i_5.agent_list = i_6.agent_list = []

        k_2 = 0
        for i_3 in _:

            k_2 += 1
            a, b = i_3
            if a.predecessor != 0:

                if a.predecessor == b:

                    b.agent_list.append(k_2)
                    b.agent_list.reverse()
                    required_tuple = ((a, b), b.agent_list, i + k)
                    for i_5, i_6 in _:
                        i_5.successor = i_6.successor = i_6.predecessor = (
                            i_5.predecessor
                        ) = 0
                        i_5.agent_list = i_6.agent_list = []

                    return required_tuple
            a.successor = b
            b.predecessor = a
            a.agent_list.append(k_2)

        for i_5, i_6 in _:
            i_5.successor = i_6.successor = i_6.predecessor = i_5.predecessor = 0
            i_5.agent_list = i_6.agent_list = []

        k_2 = 0
        for j in i_1[k]:
            j.explored = False
            j.agent_list = []

        for j in i_1[k]:

            k_2 += 1
            if not j.explored:

                j.explored = True
                j.agent_list.append(k_2)
            else:
                print("Agent_list for j = " + str(j.agent_list))
                j.agent_list.append(k_2)
                required_tuple = (j, j.agent_list, i + k)
                print(j.agent_list)
                print("Executed after edge conflict")
                for _ in i_1[k]:
                    _.explored = False
                    _.agent_list = []

                return required_tuple

        for _ in i_1[k]:
            _.explored = False
            _.agent_list = []

    return None


def number_assignments(x):

    if x == 0:
        return 1

    elif x % 2 == 0:
        return 2

    else:
        return 1

import numpy as np

def circumcircle(points):
    # Unpack the points
    A, B, C = points

    # Calculate the midpoints of AB and BC
    D = (A + B) / 2
    E = (B + C) / 2

    # Calculate the slopes of AB and BC
    slope_AB = (B[1] - A[1]) / (B[0] - A[0])
    slope_BC = (C[1] - B[1]) / (C[0] - B[0])

    # Calculate the slopes of the perpendicular bisectors
    perp_slope_AB = -1 / slope_AB
    perp_slope_BC = -1 / slope_BC

    # Equations of the perpendicular bisectors (y = mx + c)
    # y = perp_slope_AB * (x - D[0]) + D[1]
    # y = perp_slope_BC * (x - E[0]) + E[1]

    # Solving the intersection of the two lines to find the circumcenter
    def perp_bisector_eq(slope, midpoint):
        return lambda x: slope * (x - midpoint[0]) + midpoint[1]

    bisector_AB = perp_bisector_eq(perp_slope_AB, D)
    bisector_BC = perp_bisector_eq(perp_slope_BC, E)

    # Intersection of the perpendicular bisectors
    circumcenter_x = (bisector_BC(E[0]) - bisector_AB(D[0]) + perp_slope_AB * D[0] - perp_slope_BC * E[0]) / (perp_slope_AB - perp_slope_BC)
    circumcenter_y = bisector_AB(circumcenter_x)
    circumcenter = np.array([circumcenter_x, circumcenter_y])

    # Calculate the radius
    radius = np.linalg.norm(A - circumcenter)

    return circumcenter, radius

def interpolate_points_on_circle(point1, point2, center, radius, num_points = 7):
    # Convert points to numpy arrays for easier manipulation
    p1 = np.array(point1) - np.array(center)
    p2 = np.array(point2) - np.array(center)
    
    # Normalize the points to lie on the unit circle
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)
    
    # Create an array of interpolation factors between 0 and 1
    t_values = np.linspace(0, 1, num_points)
    
    # Interpolate between the normalized points
    interpolated_points = [(1 - t) * p1_norm + t * p2_norm for t in t_values]
    
    # Normalize the interpolated points and scale them to the circle's radius
    interpolated_points = [radius * p / np.linalg.norm(p) for p in interpolated_points]
    
    # Translate points back to the original center
    interpolated_points = [tuple(p + np.array(center)) for p in interpolated_points]

    interpolated_points = [(round(i[0],2),round(i[1],2)) for i in interpolated_points]
    interpolated_points = interpolated_points[1:-1]
    
    return interpolated_points

def are_collinear(p1, p2, p3, tol=1e-9):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the area of the triangle
    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    
    # Check if the area is close to zero
    return area < tol

def create_array_preliminary(array):
    
    duplicate = False
    same_tuple = False
    next_coordinate = 0
    array.append(array[-1])
    new_array = []
    for i in range(len(array[:-1])):

        if i != 0 or array[i] != array[0]:
            if array[i] == array[i + 1]:
                if array[i] == array[ - 1]:
                   
                    if not duplicate:
                        new_array.append((round((array[i-1][0] + array[i][0])/2,2),round((array[i-1][1] + array[i][1])/2,2)))
                    
                    new_array.append(array[i])
                    return new_array
               
                elif not same_tuple:
                    
                   # new_array.append((round((array[i-1][0] + array[i][0])/2,2),round((array[i-1][1] + array[i][1])/2,2)))
                    new_array.append((round((array[i][0] + array[i-1][0])/2,2),round((array[i][1] + array[i-1][1])/2,2)))
                    new_array.append(array[i])
                    same_tuple = True
                    next_coordinate = [x for x in array[i+1:] if x != array[i]][0]
                    next_coordinate = (round((array[i][0] + next_coordinate[0])/2,2),round((array[i][1] + next_coordinate[1])/2,2))
                    new_array.append(next_coordinate)


                else:

                    new_array.append(new_array[-1])
                
                    print(1)
                    print(new_array[-1])
               
            elif same_tuple:
                
                new_array.append(new_array[-1])
                print(2)
                print(new_array[-1])
               # new_array.append(array[i])
                same_tuple = False
                duplicate = True
            
            else: 
                if not duplicate:
                    new_array.append((round((array[i-1][0] + array[i][0])/2,2),round((array[i-1][1] + array[i][1])/2,2)))
                else:
                    duplicate = False
                print(3)
                print(new_array[-1])
                new_array.append(array[i])
        else:
            new_array.append(array[i])


def create_array(array):
    
    set_of_nodes_to_be_added = []
    new_array = []
    circum_coordinates = []
    for i in range(len(array)):
        
        if i == 0:
            new_array.append(array[i])
            continue

        if array[i] == array[i - 1]:
            
            new_array.extend([array[i-1] for e in range(11)])
            new_array.append(array[i])

        else:
            if circum_coordinates!= []:

                new_array.extend(circum_coordinates.pop())
                new_array.append(array[i])

            else:

                if array[i - 1] == array[0] or array[i] == array[-1]:
                    if array[i][0] != array[i - 1][0]:
                        the_slope = (array[i][1] - array[i - 1][1])/(array[i][0] - array[i - 1][0])
                        if  array[i - 1] == array[0]:
                            new_array.extend([(round(array[0][0] + (k/6)*(array[i][0] - array[i - 1][0]),2),round(array[0][1] + (k/6)*(array[i][0] - array[i - 1][0])*the_slope,2)) for k in range(1,6)])
                        else:
                            new_array.extend([(round(array[i - 1][0] + (k/6)*(array[i][0] - array[i - 1][0]),2),round(array[i - 1][1] + (k/6)*(array[i][0] - array[i - 1][0])*the_slope,2)) for k in range(1,6)])
                    else:

                        print('Over here = '+ str(array[i]))
                        if array[i - 1] == array[0]:
                            new_array.extend([(array[0][0],round(array[0][1] + (k/6)*(array[i][1] - array[i - 1][1]),2)) for k in range(1,6)])
                        else:
                            new_array.extend([(array[i - 1][0],round(array[i - 1][1] + (k/6)*(array[i][1] - array[i - 1][1]),2)) for k in range(1,6)])
                    new_array.append(array[i])
                
                else:

                    a,b = (array[i - 1],array[i])

                    for _ in array[i+1:]:

                        if _ != b:
                            c = _
                            break
                    
                    print('a,b,c = ' + str((a,b,c)))
                    if not are_collinear(a,b,c) :
                        
                        print((a,b,c))
                        circumcenter,radius = circumcircle((np.array(a),np.array(b),np.array(c)))
                        print('circumcenter = ' + str(circumcenter))
                        print('radius = ' + str(radius))
                        circum_coordinates.append(interpolate_points_on_circle(a,b,circumcenter,radius,7))
                        circum_coordinates.append(interpolate_points_on_circle(b,c,circumcenter,radius,7))
                        print('circum_coordinates = ' + str(circum_coordinates))
                        circum_coordinates.reverse()
                    else:

                        if a[0] == b[0] == c[0]:

                            circum_coordinates.append([(a[0],round(a[1] + (k/6)*(b[1] - a[1]),2)) for k in range(1,6)])
                            circum_coordinates.append([(a[0],round(b[1] + (k/6)*(c[1] - b[1]),2)) for k in range(1,6)])

                        elif a[1] == b[1] == c[1]:

                            circum_coordinates.append([(round(a[0] + (k/6)*(b[0] - a[0]),2),a[1]) for k in range(1,6)])
                            circum_coordinates.append([(round(b[0] + (k/6)*(c[0] - b[0]),2),a[1]) for k in range(1,6)])

                        else:
                            the_slope = (b[1] - a[1])/(b[0] - a[0])
                            circum_coordinates.append([(round(a[0] + (k/6)*(b[0] - a[0]),2),round(a[1] + (k/6)*(b[0] - a[0])*the_slope,2)) for k in range(1,6)])
                            the_slope = (c[1] - b[1])/(c[0] - b[0])
                            circum_coordinates.append([(round(b[0] + (k/6)*(c[0] - b[0]),2),round(b[1] + (k/6)*(c[0] - b[0])*the_slope,2)) for k in range(1,6)])
                        
                        circum_coordinates.reverse()

                    new_array.extend(circum_coordinates.pop())
                    new_array.append(array[i])
           
    return new_array
        
def give_angle(x,y):
    
    if x == (0,0) or y == (0,0) :
        return 0
    
    if x == y:
        return None

    if y[1] == x[1]:
        if y[0] > x[0]:
            return 270
        else:
            return 90
        
    elif y[0] == x[0]:
        if y[1] > x[1]:
            return 0
        else:
            return 180

    else:

        the_slope = (y[1] - x[1])/(y[0] - x[0])
        the_angle = np.arctan(the_slope)
        the_angle = np.degrees(the_angle)

        if the_angle > 0:

            if y[0] > x[0] :
                return 270 + the_angle

            elif y[0] < x[0] :
                return 90 + the_angle

            
        else:

            if y[0] > x[0]:
                return 90 + the_angle

            elif y[0] < x[0]:
                return 270 + the_angle


def plot_hex_grid(centroids, objects = None, colors = None, save_path = 'images',next_angles = [], destination_array = [],centroids_to_plot = []):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    list_of_co = [(9.5, 9.16),(11.0, 10.03), (8.0, 10.03)]
    color_list = ['red','yellow','green']
    # Plot centroids
    for centroid in centroids:
        if centroid in destination_array:
            facecolor = colors[destination_array.index(centroid)]
            hexagon = patches.RegularPolygon(centroid, numVertices = 6, radius = 1,orientation = np.pi/2 , alpha = 0.5, edgecolor='k',facecolor = facecolor)
        else:
            hexagon = patches.RegularPolygon(centroid, numVertices = 6, radius = 1,orientation = np.pi/2 , alpha = 0.5, edgecolor='k',facecolor = 'none')
        
        ax.add_patch(hexagon)

    the_empty_list = []
    for obj, color,angle in zip(objects[0], colors, next_angles):
        #centroid = centroids[obj]
        if obj == (0,0):
            continue
       # print('obj = ' + str(obj))
       # object_circle = patches.Circle((obj[0] + .2*np.cos())
        object_ellipse = patches.Ellipse(obj, width=0.2, height=0.4, angle = angle, color=color)
        if object_ellipse in the_empty_list:

            hexagon = patches.RegularPolygon(object_ellipse, numVertices = 6, radius = 1,orientation = np.pi/2 , alpha = 0.5, edgecolor='k',facecolor = 'brown')
            ax.add_patch(hexagon)
        else:
            the_empty_list.append(object_ellipse)
     
        ax.add_patch(object_ellipse)
        
    if objects[1] != []:
        for _ in range(len(objects[1]) - 1):
            
            if _ % 2 == 1:
                continue
            
            k = 0
            list__to_plot = list(zip(objects[1][_],objects[1][_ + 1]))
            for i in list__to_plot:
                if (0.0,0.0) not in i and i[0] != i[1]:
                    ax.plot([i[0][0],i[1][0]],[i[0][1],i[1][1]],color = colors[k], linestyle = '-', linewidth = 1)
                
                k += 1
    ax.autoscale_view()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to release memory
    else:
        plt.show()
