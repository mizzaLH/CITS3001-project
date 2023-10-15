import ast
import json
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

# Test the function
filename = "/Users/michaelhardie/Downloads/mario_info.txt"
data = extract_coordinates(filename)

games = [data[i:i+3] for i in range(0, len(data), 3)]
# Unpack x and y values
num_rows = len(games) // 3
num_rows += len(games) % 3  # Adjust for remainder

fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

for i, game in enumerate(games):
    row = i // 3
    col = i % 3
    x_values, y_values = zip(*game)
    
    axes[row, col].plot(x_values, y_values, marker='o', linestyle='-')
    axes[row, col].set_title(f'Game {i+1}')
    axes[row, col].grid(True)

plt.tight_layout()
plt.show()
'''
x_values, y_values = zip(*data)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, c='blue', marker='o', edgecolors='black')
plt.title('x, y Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
'''