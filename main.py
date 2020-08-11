import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter


def tyler_gram_schmidt(vectors):
    """ Orthogonalize the vectors in the columns of a matrix, using Gram-Schmidt.

    :param np.array vectors: (m x n) Each column is an m-dimensional vector.

    :return np.array orthovectors: (m x r) A new set of m-dimensional column vectors
        that are orthonormal. There are r vectors (columns)
    """

    ## Loop through the vectors and make each one orthogonal to all the previous ones.

    orthovectors = []
    for vec in vectors.T:
        # Remove the projection of this vector onto each of the previous vectors,
        # ensuring that what's left is orthogonal to all previous vectors.
        for prev_vec in orthovectors:
            vec = vec - (np.dot(vec, prev_vec) * prev_vec)
        # If the remaining vector is the 0-vector, this vector is not independent of the
        # previous ones, so it adds nothing new, and is thrown out.
        if np.allclose(vec, np.zeros(len(vec))):
            continue
        # Normalize so it's a unit vector, before storing it.
        unit_vec = vec / np.linalg.norm(vec)
        orthovectors.append(unit_vec)

    orthovectors = np.array(orthovectors).T

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
    ## Check that the output columns span the same space as the input columns.
    vectors = random_matrix(10, 4)
    # vectors = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]])
    orthovectors = tyler_gram_schmidt(vectors)
    # Make sure they have the same rank, and combining them doesn't increase that rank.
    np.testing.assert_equal(
        np.linalg.matrix_rank(orthovectors), np.linalg.matrix_rank(vectors)
    )
    np.testing.assert_equal(
        np.linalg.matrix_rank(orthovectors),
        np.linalg.matrix_rank(np.hstack((orthovectors, vectors))),
    )
    # Also make sure the vectors are orthonormal, by checking Q'Q=I.
    np.testing.assert_allclose(
        np.dot(orthovectors.T, orthovectors), np.eye(orthovectors.shape[1]), atol=1e-10
    )

    # ## Chart runtime growth of Tyler's algo and numpy's algo.
    # evaluate_gram_schmidt_runtime(
    #     builtin_orthogonalization, up_to_m=500, up_to_n=500, trials_per_mn=5
    # )
    # evaluate_gram_schmidt_runtime(
    #     tyler_gram_schmidt, up_to_m=500, up_to_n=500, trials_per_mn=5
    # )


if __name__ == "__main__":
    main()
