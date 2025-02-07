from functions import *
import os
import imageio
import pickle
import copy
with open('myobject_2.pkl','rb') as file:
     set_of_nodes = pickle.load(file)
# final_array = [[(12.5, 10.89), (0.5, 3.96), (12.5, 12.62), (0.5, 5.7), (12.5, 5.7), (0.5, 7.43), (12.5, 3.96), (0.5, 12.62), (2.0, 11.76), (0.5, 9.16), (8.0, 1.37), (8.0, 11.76), (12.5, 0.5), (3.5, 12.62), (5.0, 11.76), (3.5, 0.5), (6.5, 0.5), (2.0, 1.37)], [(11.0, 10.03), (0.5, 2.23), (11.0, 11.76), (0.5, 7.43), (11.0, 6.56), (2.0, 6.56), (12.5, 5.7), (2.0, 11.76), (2.0, 10.03), (0.5, 10.89), (8.0, 3.1), (9.5, 10.89), (11.0, 1.37), (5.0, 11.76), (5.0, 10.03), (3.5, 2.23), (5.0, 1.37), (2.0, 3.1)], [(11.0, 8.29), (2.0, 1.37), (9.5, 10.89), (2.0, 8.29), (9.5, 7.43), (3.5, 5.7), (12.5, 7.43), (3.5, 10.89), (0.5, 9.16), (2.0, 11.76), (8.0, 4.83), (11.0, 10.03), (9.5, 2.23), (6.5, 10.89), (5.0, 8.29), (5.0, 3.1), (3.5, 2.23), (2.0, 4.83)], [(9.5, 7.43), (3.5, 0.5), (9.5, 9.16), (3.5, 9.16), (8.0, 8.29), (3.5, 3.96), (12.5, 9.16), (5.0, 10.03), (0.5, 9.16), (2.0, 11.76), (6.5, 5.7), (11.0, 8.29), (8.0, 3.1), (8.0, 10.03), (3.5, 7.43), (6.5, 3.96), (2.0, 3.1), (3.5, 5.7)], [(8.0, 6.56), (5.0, 1.37), (9.5, 7.43), (3.5, 10.89), (6.5, 9.16), (5.0, 3.1), (11.0, 10.03), (5.0, 8.29), (0.5, 9.16), (2.0, 11.76), (6.5, 7.43), (11.0, 6.56), (6.5, 3.96), (9.5, 9.16), (2.0, 6.56), (8.0, 4.83), (0.5, 3.96), (5.0, 6.56)], [(6.5, 5.7), (6.5, 0.5), (8.0, 6.56), (5.0, 11.76), (5.0, 10.03), (6.5, 3.96), (9.5, 10.89), (5.0, 6.56), (0.5, 9.16), (2.0, 11.76), (5.0, 8.29), (11.0, 4.83), (5.0, 4.83), (11.0, 8.29), (0.5, 5.7), (9.5, 5.7), (0.5, 3.96), (6.5, 7.43)], [(6.5, 3.96), (6.5, 0.5), (6.5, 5.7), (5.0, 11.76), (3.5, 10.89), (8.0, 3.1), (8.0, 11.76), (5.0, 4.83), (0.5, 9.16), (2.0, 11.76), (3.5, 9.16), (12.5, 3.96), (3.5, 5.7), (12.5, 7.43), (0.5, 5.7), (11.0, 6.56), (0.5, 3.96), (8.0, 8.29)], [(5.0, 3.1), (6.5, 0.5), (5.0, 4.83), (5.0, 11.76), (3.5, 12.62), (9.5, 2.23), (8.0, 11.76), (6.5, 3.96), (0.5, 9.16), (2.0, 11.76), (2.0, 10.03), (12.5, 3.96), (2.0, 6.56), (12.5, 5.7), (0.5, 5.7), (11.0, 8.29), (0.5, 3.96), (9.5, 9.16)], [(3.5, 2.23), (6.5, 0.5), (5.0, 3.1), (5.0, 11.76), (3.5, 12.62), (11.0, 1.37), (8.0, 11.76), (6.5, 2.23), (0.5, 9.16), (2.0, 11.76), (0.5, 10.89), (12.5, 3.96), (0.5, 7.43), (12.5, 5.7), (0.5, 5.7), (11.0, 10.03), (0.5, 3.96), (11.0, 8.29)], [(2.0, 1.37), (6.5, 0.5), (5.0, 1.37), (5.0, 11.76), (3.5, 12.62), (12.5, 0.5), (8.0, 11.76), (8.0, 1.37), (0.5, 9.16), (2.0, 11.76), (0.5, 12.62), (12.5, 3.96), (0.5, 7.43), (12.5, 5.7), (0.5, 5.7), (11.0, 11.76), (0.5, 3.96), (11.0, 10.03)], [(2.0, 1.37), (6.5, 0.5), (3.5, 0.5), (5.0, 11.76), (3.5, 12.62), (12.5, 0.5), (8.0, 11.76), (8.0, 1.37), (0.5, 9.16), (2.0, 11.76), (0.5, 12.62), (12.5, 3.96), (0.5, 7.43), (12.5, 5.7), (0.5, 5.7), (12.5, 12.62), (0.5, 3.96), (12.5, 10.89)]]
#final_array = [[(12.5, 10.89), (2.0, 1.37), (2.0, 11.76), (0.5, 0.5), (12.5, 0.5), (0.5, 9.16), (5.0, 11.76), (0.5, 3.96), (5.0, 1.37), (11.0, 11.76), (12.5, 9.16), (12.5, 5.7), (0.5, 2.23), (11.0, 1.37)], [(12.5, 9.16), (0.5, 0.5), (3.5, 10.89), (0.5, 2.23), (11.0, 1.37), (2.0, 8.29), (5.0, 10.03), (0.5, 5.7), (5.0, 3.1), (12.5, 10.89), (12.5, 7.43), (11.0, 6.56), (2.0, 1.37), (11.0, 3.1)], [(12.5, 7.43), (0.5, 2.23), (5.0, 10.03), (2.0, 3.1), (11.0, 3.1), (2.0, 6.56), (5.0, 8.29), (0.5, 7.43), (5.0, 4.83), (12.5, 9.16), (11.0, 6.56), (9.5, 7.43), (2.0, 1.37), (11.0, 4.83)], [(12.5, 5.7), (0.5, 2.23), (6.5, 10.89), (3.5, 3.96), (11.0, 4.83), (2.0, 4.83), (3.5, 7.43), (2.0, 8.29), (3.5, 5.7), (12.5, 7.43), (9.5, 5.7), (8.0, 8.29), (2.0, 1.37), (11.0, 6.56)], [(12.5, 3.96), (0.5, 2.23), (8.0, 10.03), (5.0, 4.83), (11.0, 6.56), (2.0, 3.1), (2.0, 6.56), (2.0, 10.03), (3.5, 7.43), (12.5, 5.7), (8.0, 4.83), (6.5, 9.16), (2.0, 1.37), (11.0, 8.29)], [(12.5, 2.23), (0.5, 2.23), (9.5, 9.16), (6.5, 5.7), (11.0, 8.29), (3.5, 2.23), (2.0, 4.83), (3.5, 10.89), (2.0, 8.29), (12.5, 3.96), (6.5, 3.96), (5.0, 10.03), (2.0, 1.37), (11.0, 10.03)], [(11.0, 1.37), (0.5, 2.23), (9.5, 7.43), (8.0, 6.56), (11.0, 10.03), (5.0, 1.37), (0.5, 3.96), (5.0, 11.76), (0.5, 9.16), (12.5, 2.23), (5.0, 3.1), (3.5, 10.89), (2.0, 1.37), (12.5, 10.89)], [(11.0, 1.37), (0.5, 2.23), (11.0, 6.56), (9.5, 7.43), (11.0, 11.76), (5.0, 1.37), (0.5, 3.96), (5.0, 11.76), (0.5, 9.16), (12.5, 0.5), (3.5, 2.23), (2.0, 11.76), (2.0, 1.37), (12.5, 10.89)], [(11.0, 1.37), (0.5, 2.23), (12.5, 5.7), (11.0, 8.29), (11.0, 11.76), (5.0, 1.37), (0.5, 3.96), (5.0, 11.76), (0.5, 9.16), (12.5, 0.5), (2.0, 1.37), (2.0, 11.76), (2.0, 1.37), (12.5, 10.89)], [(11.0, 1.37), (0.5, 2.23), (12.5, 5.7), (12.5, 9.16), (11.0, 11.76), (5.0, 1.37), (0.5, 3.96), (5.0, 11.76), (0.5, 9.16), (12.5, 0.5), (0.5, 0.5), (2.0, 11.76), (2.0, 1.37), (12.5, 10.89)], [(11.0, 1.37), (0.5, 2.23), (12.5, 5.7), (12.5, 9.16), (11.0, 11.76), (5.0, 1.37), (0.5, 3.96), (5.0, 11.76), (0.5, 9.16), (12.5, 0.5), (0.5, 0.5), (2.0, 11.76), (2.0, 1.37), (12.5, 10.89)]]
with open("final_array.pkl", "rb") as file:
    final_array = pickle.load(file)
list_of_co = final_array[1]
final_array = final_array[0]
print(final_array)
final_array = list(final_array)
print(type(final_array))
print(final_array)
# final_array = [[(6.5, 3.96),(6.5, 5.7)],[(5.0, 4.83),(6.5, 3.96)],[(6.5, 5.7),(6.5, 3.96)]]
# final_array = [[(0.5, 0.5)], [(0.5, 2.23)], [(2.0, 3.1)],[(2.0, 3.1)], [(2.0, 4.83)]]
centroids_to_plot = []
for i in final_array:
    centroids_to_plot.extend(i)
centroids_to_plot_2 = []
for i in centroids_to_plot:
    k_2 =  set_of_nodes[1][set_of_nodes[3][i]].osf_dict
    for k in k_2:
        if k_2[k] != 0:
            centroids_to_plot_2.append(k_2[k].location)

#centroids_to_plot.extend(centroids_to_plot_2)
print('reached here')
centroids_to_plot = list(set(centroids_to_plot))
centroids_to_plot.append((0.5, 3.96))
final_array = np.array(final_array).T
final_array = [list(zip(final_array[0][i],final_array[1][i])) for i in range(len(final_array[0]))]
final_array = [i[:np.min([e for e in range(len(i)) if i[e] == i[-1]]) + 1] for i in final_array]
#print('preliminary final array = ' + str(final_array))
final_array = [create_array_preliminary(i) for i in final_array] 

print('final array preliminary = '+ str(final_array))
new_array = []
final_array = [create_array(i) for i in final_array]
print('final_array = ' + str(final_array))
#print(final_array[7])
max_length = np.max([len(i) for i in final_array])
for i in range(len(final_array)):
    final_array[i].extend([(0,0) for e in range(max_length - len(final_array[i]))])

final_array = np.array(final_array)
final_array = final_array.T
final_array = [list(zip(final_array[0][i],final_array[1][i])) for i in range(len(final_array[0]))]
centroids_to_plot.append((3.5, 10.89))
imaginary_dict = {}

# list_of_co = [(0.5, 3.96), (0.5, 2.23), (2.0, 3.1)]
# color_list = ['brown']


def plot_hex_grid(centroids, objects = None, colors = None, save_path = 'images', next_angles = [], destination_array = [], centroids_to_plot = [], imaginary_array =[],imaginary_dict = imaginary_dict,list_of_co = list_of_co):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    obstacle_color = 'brown'
    for centroid in centroids:
        if centroid in list_of_co :
            facecolor = obstacle_color
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
                
                if _ == 0:
                    if (k ,i[0]) in imaginary_array or (k ,i[1]) in imaginary_array :

                        try:
                            the_imaginary_list = imaginary_dict[(k ,i[0])]
                        except KeyError:
                            the_imaginary_list = imaginary_dict[(k ,i[1])]

                        wait = False
                        if 'wait' in the_imaginary_list:
                            wait = True
                            the_imaginary_list = the_imaginary_list[:-1]
                        for i_ in range(len(the_imaginary_list)):
                            if i_ % 3 == 0 and i_ != len(the_imaginary_list) - 1:
                                j = [the_imaginary_list[i_],the_imaginary_list[i_ + 1]]
                                if wait:
                                    ax.plot([j[0][0],j[1][0]],[j[0][1],j[1][1]],color = colors[k + 1], linestyle = ':', linewidth = 4)
                                else:
                                    ax.plot([j[0][0],j[1][0]],[j[0][1],j[1][1]],color = colors[k], linestyle = ':', linewidth = 4)
                        
                        wait = False
                k += 1
    ax.autoscale_view()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to release memory
    else:
        plt.show()

# Example list of centroids (x, y coordinates)
centroids = set_of_nodes[2]
k = 0
colors = [
    "red",
    "blue",
    "green",
    "cyan",
   "magenta",
 "yellow",
    "black","orange","purple",
    "brown","pink",
    "gray","olive",
    "teal","navy",
    "maroon", "gold","lime",
"purple",
    "brown","pink",
    "gray"
][:14]
color_dict = {}
print(colors)
next_angles_dict = {i:0 for i in range(1,15)}
for _ in range(len(final_array) - 1):
    the_last_array = []
    next_angles = list(zip(final_array[_],final_array[_ + 1]))
    next_angles = [give_angle(x,y) for x,y in next_angles]
    for i in range(len(next_angles)):
        if next_angles[i] == None:
                    next_angles[i] = next_angles_dict[i + 1]
    for i in range(len(next_angles)):
        next_angles_dict[i + 1] = next_angles[i]
    # next_angles = [(i[1][0] - i[0][0],i[1][1] - i[0][1]) for i in next_angles]
    # next_angles = [dictionary_dict[(direction(i[0]),direction(i[1]))] for i in next_angles]
    # final_array_4.pop()
    # if k != 0:
    #    objects = [final_array.pop(),final_array_3.pop()]
    # else:
    #    objects = [final_array.pop(),0]
    k += 1 
    objects = final_array[_]
    print('objects = ' + str(objects))
    plot_hex_grid(centroids,objects = [objects,final_array[_ + 1:]],colors = colors ,save_path = 'images/image' + str(0) + str(k).zfill(3) + '.png',next_angles = next_angles,destination_array = [],centroids_to_plot = centroids_to_plot)

image_dir = 'images'

# Output video file path
output_video_path = 'generated_path.mp4'

# List all PNG files in the directory
image_files = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')])
print(image_files)
# Create a writer object
writer = imageio.get_writer(output_video_path, fps = 6)  # Adjust fps as needed

# Iterate over each PNG image and add it to the video
for image_file in image_files:
    image = imageio.imread(image_file)
    writer.append_data(image)

# Close the writer
writer.close()

print("Video created successfully!")