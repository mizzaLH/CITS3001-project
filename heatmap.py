import ast
import numpy as np
import matplotlib.pyplot as plt

def extract_coordinates(filename):
    coordinates = []
    
    with open(filename, 'r') as f:
        for line in f:
            data = ast.literal_eval(line.strip())
            if data['stage'] == 1:
                x = data['x_pos']
                y = data['y_pos']
                coordinates.append((x, y))
                
    return coordinates

filename = "/Users/michaelhardie/Desktop/Uni/2023/Algorithms/mario_RB1.txt"
data = extract_coordinates(filename)

# Unpack x and y values
x_values, y_values = zip(*data)

# Create heatmap
heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=(100,100))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure(figsize=(12, 8))
plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='gray')
plt.colorbar(label="Frequency")
plt.title("Player Positions Heatmap")
plt.xlabel("x position")
plt.ylabel("y position")
plt.show()
