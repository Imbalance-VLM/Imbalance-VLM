
import numpy as np

# Read input file
with open('./logs/final_res.txt', 'r') as f:
    lines = f.readlines()

# Initialize a dictionary to store the mean values for each algorithm
means = {}

# Iterate through each line in the input file
for line in lines:
    # Split the line into algorithm name and values
    items = line.split()
    algorithm = '_'.join(items[0].split('_')[:-1])
    values = np.array(list(map(float, items[1:])))
    print(algorithm,values.size)
    # Add the values to the corresponding algorithm in the dictionary
    if algorithm not in means:
        means[algorithm] = values
    else:
        means[algorithm] += values

# Calculate the mean values for each algorithm and store them in a new file
with open('./logs/final_mean_res.txt', 'w') as f:
    for algorithm in means:
        means[algorithm] /= 3
        means_str = ' '.join([f"{x:0.2f}" for x in means[algorithm]])
        f.write(f"{algorithm} {means_str}\n")
        # Write the mean values to the output file

