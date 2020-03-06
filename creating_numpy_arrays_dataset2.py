# Date: 03/06/2020
# Author: Sierra Bonilla
# 
# Purpose: This file takes in a dictionary of DataFrames and a
# specified gene of interest and returns a plot of the edge
# convolution of the mouse coronal slice of your specified gene.
# 
# To Do: Make numpy array into X and Y coordinates, so that it will plot
# correctly. Right now, it will plot 1 dimensionally. Also, need to
# add in the correct names of the columns for function numpy_arrays
#
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import imageio
import matplotlib


def numpy_arrays(dictionary, gene):
    """
    This function takes in a dictionary of DataFrames and
    a specified gene. This function will return a numpy
    array of shape (len(dictionary),), where len(dictionary)
    is the number of hashes or X,Y coordinates. This can
    be changed later to add X,Y coordinates when known to make
    a numpy array of shape (X,Y).
    """
    array = np.zeros((len(dictionary),))
    for i in dictionary:
        count = 0
        if gene in i[*name of gene column*]: #add name of gene column
            df = i[i[*name of gene column*] == gene] #add name of gene column
            array[count] = df[*name of count column*] #add name of count column
        count += 1
    print(f'Returning Numpy Array for the Gene: {gene}')
    return array


def edge_convolution(array):
    """
    This function takes in the array of a specified gene.
    This function performed an edge convolution on the image
    to determine the greatest difference in gene expression.
    The purpose is hopefully to outline the area with the genes
    expressed highest.
    """
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel_height, kernel_width = kernel.shape
    array_height, array_width = array.shape

    result_height = array_height - kernel_height + 1
    result_width = array_width - kernel_width + 1
    result = np.zeros((result_height, result_width))
    for i in range(result_height):
        for j in range(result_width):
            curr = array[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(curr * kernel)
    return result


def plot_result_individual(edge_img, gene_name, output_file):
    """
    This function plots the individual gene convolved result.
    """
    edge_imgplot = plt.imshow(edge_img, cmap=plt.cm.gray)
    plt.set_title(f'{gene_name} Edge Convolved Image')
    plt.savefig(output_file)



def main():
    array_gene = numpy_arrays(dictionary, gene_name)
    result = edge_convolution(array_gene)
    plot_result_individual(result, gene_name, output_file='edge.png')


if __name__ == '__main__':
    main()