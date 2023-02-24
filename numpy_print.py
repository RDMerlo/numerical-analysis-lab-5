def print_array(A, message = ""):
    print(message)
    for i in range(0, len(A), 1):
        for j in range(0, len(A[i]), 1):
            print("%.4f     " % A[i][j], end="")
        print("")
    print()

def print_vector(A, message = ""):
    print(message, end="")
    for j in range(len(A)):
        print("%.4f     " % A[j], end="")
    print("\n")