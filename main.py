import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import huffmancodec as huffc

# outputs a dot-graph showing each variable relative to MPG
def data_viz(var_names, matrix):
    var_names.remove('MPG')

    # arranges the grid-like graph structure
    fig, axes = plt.subplots(3, 2)
    
    # makes each of the graphs
    for i, var in enumerate(var_names):
        row = i // 2
        col = i % 2
        axes[row, col].scatter(matrix[:, i], matrix[:, -1], 5, [(0.7, 0, 0.7)])
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('MPG')
        axes[row, col].set_title(f"MPG vs {var}")
    
    plt.tight_layout()
    plt.show()

# returns a dictionary which will have each symbol of a given matrix as a key and their occurences as the corresponding value
def count_occurrences(var_names, matrix):
    data_dict = {}

    # counts and store occurrences of given symbols
    for i, var in enumerate(var_names):
        data_dict.setdefault(var, {})
        for value in matrix[:, i]:
            if value in data_dict[var]:
                data_dict[var][value] += 1
            else:
                data_dict[var].setdefault(value, 1)
    
    return data_dict

# outputs a bar-graph showing each variables' occurences
def occurence_viz(var_names, occurrences):

    # makes each of the graphs
    for i, var in enumerate(var_names):
        keys = list(occurrences[var].keys())
        values = list(occurrences[var].values())
        plt.bar(keys, values, color = "red")
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

# returns a one-column matrix where symbol grouping (binning) has been applied
def binning(matrix, variable, interval, max_val):
    
    # iterates from 0 to the maximum value of the variable, with step interval
    for i in range(0, max_val + interval + 1, interval):
        temp_matrix = matrix[:, variable]

        # condition is a boolean matrix with temp_matrix dimensions
        condition = np.where(np.logical_and(i <= temp_matrix, temp_matrix < i + interval))

        # counts the amount of times each symbol occurs in the given interval
        counts = np.bincount(temp_matrix[condition]) if np.size(temp_matrix[condition]) > 0 else 0

        # sets every value in the interval to the value with the most occurrences
        temp_matrix[condition] = np.argmax(counts)
    
    return temp_matrix

# returns the entropy of a given matrix
def entropy(matrix) -> float:
    entropy = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)

    # applies the formula of entropy
    for j in range(len(unique_values)):
        temp = unique_counts[j] / sum(unique_counts)
        entropy -= temp * math.log2(temp)
    
    return entropy 

# returns the bits per symbol value (Huffman Coding) of a given matrix based on symbol mapping
def bits_per_symbol(matrix, map_symbols):
    bits_per_symbol = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)

    # applies the formula of bits per symbol
    for j in range(len(unique_values)):
        bits_per_symbol += map_symbols[unique_values[j]] * (unique_counts[j]/sum(unique_counts))
    
    return bits_per_symbol

# returns the variance of a given matrix based on symbol mapping and its bits per symbol value
def variance(matrix, i, map_symbols, bits_per_symbol_ar):
    variance = 0
    unique_values, unique_counts = np.unique(matrix, return_counts=True)

    # applies the formula of variance
    for j in range(len(unique_values)):
        variance += pow(bits_per_symbol_ar[i] - map_symbols[unique_values[j]], 2) * (unique_counts[j] / sum(unique_counts))
    
    return variance

# returns mutual information between two variables
def mutual_information(matrix, alphabet, var1, var2):
    mi = 0
    mpg_max = alphabet[var2][-1]
    var_max = alphabet[var1][-1]
    size = np.shape(matrix)[0]

    # counts occurrences of each combination of symbols in a two-dimensional matrix
    prob_matrix = np.zeros_like(np.arange((mpg_max + 1) * (var_max + 1)).reshape((mpg_max + 1), (var_max + 1)))
    for j in range(size): 
        prob_matrix[matrix[j, var2], matrix[j, var1]] += 1

    # applies the formula of mutual information
    for j in range(mpg_max + 1):
        for k in range(var_max + 1): 
            probability = prob_matrix[j, k] / size
            mi += probability * math.log2(probability / (sum(prob_matrix[j, :]) * sum(prob_matrix[:, k] / pow(size, 2)))) if probability > 0 else 0
    
    return mi

def mi_info(val1, val2, val3, val4, val5, val6, real_mat):
    pred_mat = np.zeros_like(real_mat[:, 6]).astype(np.float32)
    diff_mat = np.zeros_like(real_mat[:, 6]).astype(np.float32)

    # applies the given formula for the predicted MPG value
    for i in range(np.shape(real_mat)[0]): 
        pred_mat[i] = - 5.5241 + val1 * real_mat[i, 0] + val2 * real_mat[i, 1] + val3 * real_mat[i, 2] + val4 * real_mat[i, 3] + val5 * real_mat[i, 4] + val6 * real_mat[i, 5]
        diff_mat[i] = real_mat[i, 6] - pred_mat[i]
    
    # prints info on both MPG values
    print(f"Real MPG Avg: {real_mat[:, 6].mean()}\nPredicted MPG Avg: {pred_mat.mean()}\nDifference Avg: {diff_mat.mean()}\nAbsolute Difference Avg: {np.absolute(diff_mat).mean()}\n\n" +
          f"Real MPG: [{' '.join(map(lambda e: str(e), real_mat[:5, 6]))} ... {' '.join(map(lambda e: str(e), real_mat[-5:, 6]))}]\n" +
          f"Predicted MPG: [{' '.join(map(lambda e: str(e), pred_mat[:5]))} ... {' '.join(map(lambda e: str(e), pred_mat[-5:]))}]\n" +
          f"Difference: [{' '.join(map(lambda e: str(e), diff_mat[:5]))} ... {' '.join(map(lambda e: str(e), diff_mat[-5:]))}]")


def main():
    # Exercise 1: loading the dataset
    # data = pd.read_excel('/home/tiago/Desktop/TI/trabalho_tp1/TP1_TI/' + 'CarDataset.xlsx')
    # data = pd.read_excel('C:\\Users\\Utilizador\\Desktop\\lei\\2 ano\\1 semestre\\TI\\TP1_TI' + 'CarDataset.xlsx')
    data = pd.read_excel('C:\\Users\\Utilizador\\Downloads\\' + 'CarDataset.xlsx')
    matrix = data.to_numpy()
    var_names = data.columns.values.tolist()

    # Exercise 2: data visualization
    data_viz(var_names, matrix)

    # Exercise 3: typecasting the matrix and defining the dataset's alphabet
    matrix = matrix.astype(np.uint16)
    alphabet_ar = []
    for i in range(np.shape(matrix)[1]): alphabet_ar.append(tuple(range(np.max(matrix[:, i]) + 1)))

    # Exercise 4: counting occurrences of symbols for each variable
    occurrences = count_occurrences(var_names, matrix)

    # Exercise 5: visualizing occurences of symbols for each variable
    occurence_viz(var_names, occurrences)

    # Exercise 6: binning operation and visualizing binned occurrences of symbols for each variable
    vars = [2, 3, 5]
    bin_matrix = data.to_numpy()
    for var in vars:
        max_val = alphabet_ar[var][-1]
        bin_matrix[:, var] = binning(bin_matrix, var, 5, max_val) if (var != 5) else binning(bin_matrix, var, 40, max_val)
    bin_occurrences = count_occurrences(var_names, bin_matrix)
    occurence_viz(var_names, bin_occurrences)

    # Exercise 7: calculating and printing entropy for each variable
    for i, var in enumerate(var_names):
        print(f"Entropy ({var}): {entropy(bin_matrix[:, i])}")
    overall_entropy = entropy(bin_matrix.flatten())
    print(f"Overall entropy: {overall_entropy}\n") 

    # Exercise 8: calculating and printing bits per symbol and variance for each variable
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
    print()
    
    # Exercise 9: calculating and printing correlation coefficients for each variable
    for i in range(6): 
        print(f"Correlation Coeficient (MPG / {var_names[i]}): {np.corrcoef(bin_matrix[:, i], bin_matrix[:, 6], rowvar=True)[0, 1]}")
    print()

    # Exercise 10: calculating and printing mutual information for each variable
    for i in range(6):
        mi = mutual_information(bin_matrix, alphabet_ar, i, 6)
        print(f"Mutual Information (MPG / {var_names[i]}): {mi}")

    #Exercise 11: calculating and printing predicted mpg value comparisons (3 scenarios)
    print("\n> Regular Values\n")
    mi_info(- 0.146, - 0.4909, 0.0026, - 0.0045, 0.6725, - 0.0059, bin_matrix)

    print("\n> Removing the variable with the least MI\n")
    mi_info(0, - 0.4909, 0.0026, - 0.0045, 0.6725, - 0.0059, bin_matrix)

    print("\n> Removing the variable with the most MI\n")
    mi_info(- 0.146, - 0.4909, 0.0026, - 0.0045, 0.6725, 0, bin_matrix)


if __name__ == "__main__":
    main()