"""
    这里是一些网上提供的核函数
    摘自：github simple-mkl-python-master
"""
import numpy as np


# Gets an array of all kernel matrices for each kernel
def get_all_kernels(X, kernel_functions):
    n = X.shape[0]
    M = len(kernel_functions)

    # Initialize array that will store all the kernels
    kernel_matrices = []

    # Loops through all kernel functions
    for m in range(M):
        kernel_func = kernel_functions[m]
        kernel_matrices.append(np.empty((n, n)))

        # Creates kernel matrix
        print('计算核矩阵')
        for i in range(n):
            for j in range(n):
                kernel_matrices[m][i, j] = kernel_func(X[i], X[j])
                # print(kernel_matrices)

    # Returns all kernel matrices
    return kernel_matrices


# Gets the combined kernel matrix
def get_combined_kernel(kernel_matrices, weights):
    M = len(kernel_matrices)
    n_width = kernel_matrices[0].shape[0]
    n_height = kernel_matrices[0].shape[1]

    combined_kernel_matrix = np.zeros((n_width, n_height))

    for m in range(M):
        combined_kernel_matrix += kernel_matrices[m] * weights[m]

    return combined_kernel_matrix


def get_combined_kernel_function(kernel_functions, weights):
    M = len(kernel_functions)

    def combined_kernel(u, v):
        result = 0
        for m in range(M):
            result += kernel_functions[m](u, v) * weights[m]
        return result

    return combined_kernel


# 线性核函数
def create_linear_kernel(u, v):
    """ Returns inner product （內积）of u and v. """
    # print(np.inner(u, v))
    return np.inner(u, v)


# 多项式核函数
def create_poly_kernel(degree, gamma, intercept=0.0):
    """ Returns polynomial kernel of specified degree and coeff gamma. """

    def poly_kernel_func(u, v):
        return (gamma*np.inner(u, v) + intercept) ** degree

    return poly_kernel_func


# gaussian/rbf 核函数
def create_rbf_kernel(gamma):
    """ Returns the gaussian/rbf kernel with specified gamma. """

    def rbf_kernel_func(u, v):
        return np.exp(-gamma*np.sum(np.abs(u - v) ** 2))

    return rbf_kernel_func


# Exponential函数
def create_exponential_kernel(gamma):
    """ Returns the exponential kernel with specified gamma. """

    def exponential_kernel_func(u, v):
        return np.exp(-gamma*np.sum(np.abs(u - v)))

    return exponential_kernel_func


# sigmoid 核函数
def create_sigmoid_kernel(gamma, intercept=0.0):
    """ Returns the sigmoid/tanh kernel with specified gamma. """

    def sigmoid_kernel_func(u, v):
        return np.arctan(gamma*np.inner(u, v) + intercept)

    return sigmoid_kernel_func


# 直方图交叉核函数
def create_histogram_kernel(u, v):
    return np.minimum(u, v).sum()