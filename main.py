import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Exercise 1 
#data = pd.read_excel('/home/tiago/Desktop/TI/trabalho_tp1/TP1_TI/' + 'CarDataset.xlsx')
#data = pd.read_excel('C:\\Users\\Utilizador\\Desktop\\lei\\2 ano\\1 semestre\\TI\\TP1_TI' + 'CarDataset.xlsx')
data = pd.read_excel('C:\\Users\\Utilizador\\Downloads\\' + 'CarDataset.xlsx')
matrix = data.to_numpy()
var_names = data.columns.values.tolist()

# Exercise 2
def data_viz(var_names, matrix):
    vars = var_names.copy()
    vars.remove('MPG')
    fig, axes = plt.subplots(3, 2)
    
    for i, var in enumerate(vars):
        row = i // 2
        col = i % 2
        axes[row, col].scatter(matrix[:, i], matrix[:, -1], 5, [(0.7, 0, 0.7)])
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('MPG')
        axes[row, col].set_title(f"MPG vs {var}")
    
    plt.tight_layout()
    plt.show()

# Exercise 3
matrix = matrix.astype(np.uint16)
# Given that the data is of type uint16, the dictionary for the dataset is every value from 0 to 65535
alphabet = tuple(range(65535))

# Exercise 4
def count_occurrences(var_names, matrix):
    data_dict = {}
    for i, var in enumerate(var_names):
        data_dict.setdefault(var, {})
        for value in matrix[:, i]:
            if value in data_dict[var]:
                data_dict[var][value] += 1
            else:
                data_dict[var].setdefault(value, 1)
    return data_dict

# Exercise 5
def occurence_viz(var_names, occurrences):
    for i, var in enumerate(var_names):
        keys = list(occurrences[var].keys())
        values = list(occurrences[var].values())
        plt.bar(keys, values, color = "red")
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

# Exercise 6
def binning(matrix, variable, interval, min_val, max_val):
    for i in range(min_val, max_val + 1, interval):
        temp_matrix = matrix[:, variable]
        condition = np.where(np.logical_and(i <= temp_matrix, temp_matrix <= i + interval))
        counts = np.bincount(temp_matrix[condition]) if np.size(temp_matrix[condition]) > 0 else 0
        temp_matrix[condition] = np.argmax(counts)
    return temp_matrix

#Exercise 7
def entropy(matrix) -> float:
    entropy = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)
    for j in range(len(unique_values)):
        temp = unique_counts[j] / sum(unique_counts)
        entropy -= temp * math.log2(temp)
    return entropy






data_viz(var_names, matrix)
occurrences = count_occurrences(var_names, matrix)
occurence_viz(var_names, occurrences)

vars = [2, 3, 5]
for var in vars:
    min_val = min(np.array(matrix[:, var]).flatten())
    max_val = max(np.array(matrix[:, var]).flatten())
    matrix[:, var] = binning(matrix, var, 5, min_val, max_val) if (var != 5) else binning(matrix, var, 40, min_val, max_val)

occurrences = count_occurrences(var_names, matrix)
occurence_viz(var_names, occurrences)

print(var_names)
matrix = data.to_numpy()
entropy_values = []

for i, var in enumerate(var_names):
        entropy_values.append(entropy(matrix[:, i]))
        print(f"Valor da entropia de {var}: {entropy(matrix[:, i])}")

overall_entropy = entropy(matrix.flatten())
print(f"Overall entropy: {overall_entropy}")        

print(entropy_values)