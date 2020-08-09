import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter


def tyler_gram_schmidt(vectors):
    """ Orthogonalize the vectors in the columns of a matrix, using Gram-Schmidt.

    :param ndarray vectors: (m x n) Each column is an m-dimensional vector.

    :return ndarray orthovectors: (m x n) A new set of m-dimensional column vectors that
        are orthonormal. (None if vectors are dependent)
    """
    ## TODO: implement this
    orthovectors = vectors
    return orthovectors


def builtin_orthogonalization(vectors):
    """ Solve the equation Ax=b by elimination.

    :param ndarray vectors: (m x n) Each column is an m-dimensional vector.

    :return ndarray orthovectors: (m x n) A new set of m-dimensional column vectors that
        are orthonormal. (None if vectors are dependent)
    """
    Q, _ = np.linalg.qr(vectors)
    return Q


def random_matrix(m, n, bounds=(-100, 100)):
    """ Generate a matrix with all random elements.

    :param int m: Number of rows of the matrix to create.
    :param int n: Number of columns of the matrix to create.
    :param (float, float) bounds: Lower and upper bound the random values are in.

    :return ndarray:
    """
    # For each value, get a random value between 0 and 1, scale it to the size of the
    # bounds, then shift it to start at the lower bound.
    # E.g. For bounds (-100, 100), a random number is between 0 and 1, then multiplied
    # so it's between 0 and 200, then shifted so it's between -100 and 100.
    return np.random.rand(m, n) * (bounds[1] - bounds[0]) + bounds[0]


def evaluate_gram_schmidt_runtime(algo, up_to_m=100, up_to_n=100, trials_per_mn=10):
    """ Time the runtime of the elimination solver as the size of the problem increases,
        and chart the result.

    :param func(vectors)->orthovectors algo: Function that takes a set of vectors as an
        ndarray (m x n) and returns a set of orthonormal vectors that span the same
        space as an ndarray (m x n)
    :param int up_to_m: Time the algo for dimensions up to this.
    :param int up_to_n: Time the algo for dimensions up to this.
    :param int trials_per_mn: Number of solves at each matrix size, to increase accuracy.
    """

    ## Run the algorithm at various vector dimensions, with the most number of vectors.

    dimensions = range(2, up_to_m, int(up_to_m / 10))

    avg_runtimes_by_dimension = []
    for m in dimensions:
        runtimes = []
        for _ in range(trials_per_mn):
            # Create some random inputs.
            vectors = random_matrix(m, up_to_n)
            # Perform the work, and measure its runtime.
            start = perf_counter()
            algo(vectors)
            end = perf_counter()
            # Log the runtime.
            runtime = end - start
            runtimes.append(runtime)
        # Log the average runtime for this dimensions value.
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_runtimes_by_dimension.append(avg_runtime)

    ## Chart the runtimes to guess the algorithm's complexity, in terms of dimensions.

    # Include a best fit polynomial to help guess the algorithm's complexity.
    # (We actually know the complexity should be O(n^_).)
    best_fit_coeffs = np.polyfit(dimensions, avg_runtimes_by_dimension, 5)
    best_fit_poly = np.poly1d(best_fit_coeffs)
    best_fit_x = np.linspace(dimensions[0], dimensions[-1])
    best_fit_y = best_fit_poly(best_fit_x)
    # Show the chart.
    fig, ax = plt.subplots()
    ax.plot(dimensions, avg_runtimes_by_dimension, "o", best_fit_x, best_fit_y)
    plt.show()

    ## Run the algorithm at various numbers of vectors, with the most dimensions.

    vector_numbers = range(2, up_to_n, int(up_to_n / 10))

    avg_runtimes_by_num_vectors = []
    for n in vector_numbers:
        runtimes = []
        for _ in range(trials_per_mn):
            # Create some random inputs.
            vectors = random_matrix(up_to_m, n)
            # Perform the work, and measure its runtime.
            start = perf_counter()
            algo(vectors)
            end = perf_counter()
            # Log the runtime.
            runtime = end - start
            runtimes.append(runtime)
        # Log the average runtime for this dimensions value.
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_runtimes_by_num_vectors.append(avg_runtime)

    ## Chart the runtimes to guess the algorithm's complexity, in terms of dimensions.

    # Include a best fit polynomial to help guess the algorithm's complexity.
    # (We actually know the complexity should be O(n^_).)
    best_fit_coeffs = np.polyfit(dimensions, avg_runtimes_by_num_vectors, 5)
    best_fit_poly = np.poly1d(best_fit_coeffs)
    best_fit_x = np.linspace(dimensions[0], dimensions[-1])
    best_fit_y = best_fit_poly(best_fit_x)
    # Show the chart.
    fig, ax = plt.subplots()
    ax.plot(vector_numbers, avg_runtimes_by_num_vectors, "o", best_fit_x, best_fit_y)
    plt.show()


def main():
    # Check that Tyler's solver gets the same answer as numpy's, on some random problem.
    vectors = random_matrix(10, 10)
    orthovectors_0 = tyler_gram_schmidt(vectors)
    orthovectors_1 = builtin_orthogonalization(vectors)
    np.testing.assert_allclose(orthovectors_0, orthovectors_1)

    # Chart runtime growth of Tyler's algo and numpy's algo.
    evaluate_gram_schmidt_runtime(
        builtin_orthogonalization, up_to_m=10, up_to_n=10, trials_per_mn=5
    )
    evaluate_gram_schmidt_runtime(
        tyler_gram_schmidt, up_to_m=10, up_to_n=10, trials_per_mn=5
    )


if __name__ == "__main__":
    main()
