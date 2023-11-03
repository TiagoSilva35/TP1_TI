import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import huffmancodec as huffc

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

def occurence_viz(var_names, occurrences):
    for i, var in enumerate(var_names):
        keys = list(occurrences[var].keys())
        values = list(occurrences[var].values())
        plt.bar(keys, values, color = "red")
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

def binning(matrix, variable, interval, min_val, max_val):
    for i in range(0, max_val + interval + 1, interval):
        temp_matrix = matrix[:, variable]
        condition = np.where(np.logical_and(i <= temp_matrix, temp_matrix < i + interval))
        counts = np.bincount(temp_matrix[condition]) if np.size(temp_matrix[condition]) > 0 else 0
        temp_matrix[condition] = np.argmax(counts)
    return temp_matrix

def entropy(matrix) -> float:
    entropy = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)
    for j in range(len(unique_values)):
        temp = unique_counts[j] / sum(unique_counts)
        entropy -= temp * math.log2(temp)
    return entropy 

def bits_per_symbol(matrix, map_symbols):
    bits_per_symbol = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)
    for j in range(len(unique_values)):
        bits_per_symbol += map_symbols[unique_values[j]] * (unique_counts[j]/sum(unique_counts))
    return bits_per_symbol

def variance(matrix, i, map_symbols, bits_per_symbol_ar):
    variance = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)
    for j in range(len(unique_values)):
        variance += pow(bits_per_symbol_ar[i] - map_symbols[unique_values[j]], 2) * (unique_counts[j] / sum(unique_counts))
    return variance

def mutual_information(matrix, var1, var2):
    mi = 0
    mpg_max = max(matrix[:, var2])
    var_max = max(matrix[:, var1])
    size = np.shape(matrix)[0]

    prob_matrix = np.zeros_like(np.arange((mpg_max + 1) * (var_max + 1)).reshape((mpg_max + 1), (var_max + 1)))
    for j in range(size): 
        prob_matrix[matrix[j, 6], matrix[j, var1]] += 1

    for j in range(mpg_max + 1):
        for k in range(var_max + 1): 
            probability = prob_matrix[j, k] / size
            mi += probability * math.log2(probability / (sum(prob_matrix[j, :]) * sum(prob_matrix[:, k] / pow(size, 2)))) if probability > 0 else 0
    return mi



def main():
    # Exercise 1 
    # data = pd.read_excel('/home/tiago/Desktop/TI/trabalho_tp1/TP1_TI/' + 'CarDataset.xlsx')
    # data = pd.read_excel('C:\\Users\\Utilizador\\Desktop\\lei\\2 ano\\1 semestre\\TI\\TP1_TI' + 'CarDataset.xlsx')
    data = pd.read_excel('C:\\Users\\Utilizador\\Downloads\\' + 'CarDataset.xlsx')
    matrix = data.to_numpy()
    var_names = data.columns.values.tolist()

    # Exercise 2 
    data_viz(var_names, matrix)

    # Exercise 3
    matrix = matrix.astype(np.uint16)
    # Given that the data is of type uint16, the dictionary for the dataset is every value from 0 to 65535
    alphabet = tuple(range(65535))

    # Exercise 4
    occurrences = count_occurrences(var_names, matrix)

    # Exercise 5
    occurence_viz(var_names, occurrences)

    # Exercise 6
    vars = [2, 3, 5]
    bin_matrix = data.to_numpy()
    for var in vars:
        min_val = min(np.array(bin_matrix[:, var]).flatten())
        max_val = max(np.array(bin_matrix[:, var]).flatten())
        bin_matrix[:, var] = binning(bin_matrix, var, 5, min_val, max_val) if (var != 5) else binning(bin_matrix, var, 40, min_val, max_val)
    bin_occurrences = count_occurrences(var_names, bin_matrix)
    occurence_viz(var_names, bin_occurrences)

    # Exercise 7
    entropy_values = []
    for i, var in enumerate(var_names):
        entropy_values.append(entropy(bin_matrix[:, i]))
        print(f"Entropy ({var}): {entropy(bin_matrix[:, i])}")
    overall_entropy = entropy(bin_matrix.flatten())
    print(f"Overall entropy: {overall_entropy}") 

    # Exercise 8
    variance_ar = []
    bits_per_symbol_ar = []

    for i, var in enumerate(var_names):
        codec = huffc.HuffmanCodec.from_data(bin_matrix[:, i])
        symbols, lengths = codec.get_code_len()
        map_symbols = {}
        for j in range(len(symbols)):
            map_symbols[symbols[j]] = lengths[j]
        bits_per_symbol_ar.append(bits_per_symbol(bin_matrix[:, i], map_symbols))
        variance_ar.append(variance(bin_matrix[:, i], i, map_symbols, bits_per_symbol_ar))
        print(f"Bits per symbol ({var}): {bits_per_symbol_ar[i]}")
        print(f"Variance ({var}): {variance_ar[i]}")
    
    # Exercise 9
    for i in range(6): 
        print(f"Correlation Coeficient (MPG / {var_names[i]}): {np.corrcoef(bin_matrix[:, i], bin_matrix[:, 6], rowvar=True)[0, 1]}")

    # Exercise 10
    for i in range(6):
        mi = mutual_information(bin_matrix, i, 6)
        print(f"Mutual Information (MPG / {var_names[i]}): {mi}")

    #Exercise 11
    pred_mpg = np.zeros_like(bin_matrix[:, 6])
    mpg_diff = np.zeros_like(bin_matrix[:, 6])
    mpg_diff = mpg_diff.astype(np.int16)

    print("\nMPG / Predicted / Difference")
    for i in range(np.shape(bin_matrix)[0]): 
        pred_mpg[i] = - 5.5241 - 0.146 * bin_matrix[i, 0] - 0.4909 * bin_matrix[i, 1] + 0.0026 * bin_matrix[i, 2] - 0.0045 * bin_matrix[i, 3] + 0.6725 * bin_matrix[i, 4] - 0.0059 * bin_matrix[i, 5]
        mpg_diff[i] = bin_matrix[i, 6] - pred_mpg[i]
    print(bin_matrix[:, 6], "\n\n", pred_mpg[:], "\n\n", mpg_diff[:])

    print("\nRemoving the variable with the least MI")
    for i in range(np.shape(bin_matrix)[0]):
        pred_mpg[i] += 0.146 * bin_matrix[i, 0]
        mpg_diff[i] = bin_matrix[i, 6] - pred_mpg[i]
    print(bin_matrix[:, 6], "\n\n", pred_mpg[:], "\n\n", mpg_diff[:])

    print("\nRemoving the variable with the most MI")
    for i in range(np.shape(bin_matrix)[0]):
        pred_mpg[i] += - 0.146 * bin_matrix[i, 0] + 0.0059 * bin_matrix[i, 5]
        mpg_diff[i] = bin_matrix[i, 6] - pred_mpg[i]
    print(bin_matrix[:, 6], "\n\n", pred_mpg[:], "\n\n", mpg_diff[:])


if __name__ == "__main__":
    main()