For testing, there are two python scripts - testing_2D.py & testing_3D.py. 

The 2D script runs the CBS algorithms on any problem instance containing hexagonal 2D grids. - 


The 2D script runs the CBS algorithm on any problem instance containing cubical 3D grids. 

In terminal - 

Just enter the directory and run the python scripts, no special libraries required, only numpy. 

In python scripts you have the option to either manually enter the indices of coordinates of the starting and ending locations for each agent within the problem. Note that as we increase the number of agents, the complexity of the problem rises - 
If you don't want to manually enter the coordinates, you can simply enter the number of agents and the script will run a random CBS problem involving that number of agents. However, note that in this case the number of agents should be even.

Outputs -  The output is a 2D array containing the locations of the agents at each timestep according to the solution as well as its transpose.

Add Ons - Manually one by one, you can add the indices for the nodes that you would like to mark as obstacles.

Additional Outputs -  Running the plot.py file right after running the testing_2D.py shall provide the simulation for that particular instance. Also, note that now both, tesing_2D.py & testing_3D.py give dictionaries containing trajectory lengths and minimum distance from the obstacles.
The simulation output will be stored within the videos directory.
Additional Instructions - Kindly delete all the images in the images folder or store them elsewhere before generating plot for another problem.
The path for image and video directory is present in plot.py file in line 173 and line 176 respectively. 
The dictionaries linking indices to coordinates is given below for each 2D and 3D case - 

2D Case = {(0.5, 0.5): 0,
 (0.5, 2.23): 1,
 (0.5, 3.96): 2,
 (0.5, 5.7): 3,
 (0.5, 7.43): 4,
 (0.5, 9.16): 5,
 (0.5, 10.89): 6,
 (0.5, 12.62): 7,
 (2.0, 11.76): 8,
 (2.0, 10.03): 9,
 (2.0, 8.29): 10,
 (2.0, 6.56): 11,
 (2.0, 4.83): 12,
 (2.0, 3.1): 13,
 (2.0, 1.37): 14,
 (3.5, 0.5): 15,
 (3.5, 2.23): 16,
 (3.5, 3.96): 17,
 (3.5, 5.7): 18,
 (3.5, 7.43): 19,
 (3.5, 9.16): 20,
 (3.5, 10.89): 21,
 (3.5, 12.62): 22,
 (5.0, 11.76): 23,
 (5.0, 10.03): 24,
 (5.0, 8.29): 25,
 (5.0, 6.56): 26,
 (5.0, 4.83): 27,
 (5.0, 3.1): 28,
 (5.0, 1.37): 29,
 (6.5, 0.5): 30,
 (6.5, 2.23): 31,
 (6.5, 3.96): 32,
 (6.5, 5.7): 33,
 (6.5, 7.43): 34,
 (6.5, 9.16): 35,
 (6.5, 10.89): 36,
 (6.5, 12.62): 37,
 (8.0, 11.76): 38,
 (8.0, 10.03): 39,
 (8.0, 8.29): 40,
 (8.0, 6.56): 41,
 (8.0, 4.83): 42,
 (8.0, 3.1): 43,
 (8.0, 1.37): 44,
 (9.5, 0.5): 45,
 (9.5, 2.23): 46,
 (9.5, 3.96): 47,
 (9.5, 5.7): 48,
 (9.5, 7.43): 49,
 (9.5, 9.16): 50,
 (9.5, 10.89): 51,
 (9.5, 12.62): 52,
 (11.0, 11.76): 53,
 (11.0, 10.03): 54,
 (11.0, 8.29): 55,
 (11.0, 6.56): 56,
 (11.0, 4.83): 57,
 (11.0, 3.1): 58,
 (11.0, 1.37): 59,
 (12.5, 0.5): 60,
 (12.5, 2.23): 61,
 (12.5, 3.96): 62,
 (12.5, 5.7): 63,
 (12.5, 7.43): 64,
 (12.5, 9.16): 65,
 (12.5, 10.89): 66,
 (12.5, 12.62): 67}

 3D case = {(0.5, 0.5, 0.5): 0,                               (3X3 cube)
 (0.5, 1.5, 0.5): 1,
 (0.5, 2.5, 0.5): 2,
 (1.5, 2.5, 0.5): 3,
 (1.5, 1.5, 0.5): 4,
 (1.5, 0.5, 0.5): 5,
 (2.5, 0.5, 0.5): 6,
 (2.5, 1.5, 0.5): 7,
 (2.5, 2.5, 0.5): 8,
 (2.5, 2.5, 1.5): 9,
 (2.5, 1.5, 1.5): 10,
 (2.5, 0.5, 1.5): 11,
 (1.5, 0.5, 1.5): 12,
 (1.5, 1.5, 1.5): 13,
 (1.5, 2.5, 1.5): 14,
 (0.5, 2.5, 1.5): 15,
 (0.5, 1.5, 1.5): 16,
 (0.5, 0.5, 1.5): 17,
 (0.5, 0.5, 2.5): 18,
 (0.5, 1.5, 2.5): 19,
 (0.5, 2.5, 2.5): 20,
 (1.5, 2.5, 2.5): 21,
 (1.5, 1.5, 2.5): 22,
 (1.5, 0.5, 2.5): 23,
 (2.5, 0.5, 2.5): 24,
 (2.5, 1.5, 2.5): 25,
 (2.5, 2.5, 2.5): 26}






