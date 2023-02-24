def print_array(A):
    for i in range(0, len(A), 1):
        for j in range(0, len(A[i]), 1):
            print("%.4f     " % A[i][j], end="")
        print("")


def print_vector(A):
    for j in range(0, len(A), 1):
        print("%.4f     " % A[j], end="")
    print("")