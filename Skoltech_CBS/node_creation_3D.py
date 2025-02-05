from functions import *
from classes_correction import *
import pickle
import sys

sys.setrecursionlimit(10000)

k = generate_grid_hexagon_3D((0.5, 0.5,0.5), [(0, 3), (0, 3),(0, 3)])
c = 1
for i in k[2]:
    _ = 0
    for j in np.array([i]) + np.array(
        [(0, c,0),(0,-c,0),(c,0,0),(-c,0,0),(0,0,-c),(0,0,c),(c,c,c),(c,-c,c),(-c,-c,c),(-c,-c,-c),(c,-c,-c),(c,c,-c),(-c,c,-c),(-c,c,c)]
    ):
        _ += 1
        if in_middle_3_D(j, dimension_extremes=[(0, 3), (0, 3),(0,3)]):
            if tuple(j) not in k[2]:

                j_2 = [(x, y,z) for x, y,z in k[2] if np.linalg.norm((x, y, z) - j) <= 1][0]
                k[1][k[2].index(i)].osf_dict[_] = k[1][k[2].index(tuple(j_2))]
            else:

                k[1][k[2].index(i)].osf_dict[_] = k[1][k[2].index(tuple(j))]
        
        else:

                k[1][k[2].index(i)].osf_dict[_] = 0

for i in k[1]:
    x, y, z = i.location
    i.location = (x, round(y, 2), z)

for i in range(len(k[2])):
    x, y, z = k[2][i]
    k[2][i] = (x, round(y, 2), z)

new_dict = {}
for i in k[1]:
    new_dict[i.location] = k[2].index(i.location)

k = list(k)
print('length of  k = ' + str(len(k)))
k.append(new_dict)
file_path = "myobject_3D_diagonal.pkl"
with open(file_path, "wb") as file:
    pickle.dump(k, file)
