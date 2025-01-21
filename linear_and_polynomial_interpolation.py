import numpy as np
# colors for the output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    MAGENTA = '\033[35m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ORANGE = '\033[38;5;208m'
    GOLD = '\033[38;5;214m'
    GRAY = '\033[38;5;243m'
PI = 3.141592653589793

def MaxNorm(matrix):
    """
    Function for calculating the max-norm of a matrix
    :param matrix: A square matrix (list of lists).
    :return: The maximum norm value of the matrix.
    """
    max_norm = 0
    for i in range(len(matrix)):
        norm = 0
        for j in range(len(matrix)):
            # Sum of organs per line with absolute value
            norm += abs(matrix[i][j])
        # Maximum row amount
        if norm > max_norm:
            max_norm = norm

    return max_norm

def Determinant(matrix, mul):
    """
    Calculate the determinant of a square matrix using recursion.

    :param matrix: A square nxn matrix (list of lists).
    :param mul: A multiplier to track recursive contributions.
    :return: The determinant value of the matrix.
    """
    width = len(matrix)
    # Stop Conditions
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            # Change the sign of the multiply number
            sign *= -1
            #  Recursive call for determinant calculation
            det = det + mul * Determinant(m, sign * matrix[0][i])
    return det

def MakeIMatrix(cols, rows):
    """"Initialize a identity matrix"""
    return [[1 if x == y else 0 for y in range(cols)] for x in range(rows)]

def MulMatrixVector(matrix, b_vector):
    """
    Function for multiplying a vector matrix
    :param matrix: Matrix nxn
    :param b_vector: Vector n
    :return: Result vector
    """
    n = len(matrix)
    result = [0] * n  # Initialize result vector with zeros

    for i in range(n):
        for j in range(len(b_vector)):
            result[i] += matrix[i][j] * b_vector[j]

    return result
    # return np.dot(matrix, b_vector) this is what it does

def MultiplyMatrix(matrixA, matrixB):
    """
    Multiply two square matrices together.

    :param matrixA: First square matrix.
    :param matrixB: Second square matrix.
    :return: The resulting matrix after multiplication.
    """
    # result matrix initialized as singularity matrix
    result = [[0 for y in range(len(matrixB[0]))] for x in range(len(matrixA))]
    for i in range(len(matrixA)):
        # iterate through columns of Y
        for j in range(len(matrixB[0])):
            # iterate through rows of Y
            for k in range(len(matrixB)):
                result[i][j] += matrixA[i][k] * matrixB[k][j]
    return result
    # return np.dot(matrixA, matrixB) this is what it does

def InverseMatrix(matrix,vector):
    """
    Compute the inverse of a square matrix.

    :param matrix: A square matrix.
    :param vector: A vector (not used in computation).
    :return: The inverse of the matrix if invertible, otherwise None.
    """
    if Determinant(matrix, 1) == 0:
        print("Error,Singular Matrix\n")
        return
    return np.linalg.inv(matrix)

def Cond(matrix, invert):
    """
    :param matrix: Matrix nxn
    :param invert: Inverted matrix
    :return: CondA = ||A|| * ||A(-1)||
    """
    print("|| A || max = ", MaxNorm(matrix))
    print("|| A(-1) || max = ", MaxNorm(invert))
    return MaxNorm(matrix)*MaxNorm(invert)

def RowXchageZero(matrix,vector):
    """
    Function for replacing rows with both a matrix and a vector
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Replace rows after a pivoting process
    """

    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            # The pivot member is not zero
            if matrix[i][i] == 0:
                temp = matrix[j]
                temp_b = vector[j]
                matrix[j] = matrix[i]
                vector[j] = vector[i]
                matrix[i] = temp
                vector[i] = temp_b

    return [matrix, vector]

def RowXchange(matrix, vector):
    """
    Perform row swapping to ensure numerical stability during pivoting.

    :param matrix: A square matrix nxn.
    :param vector: A corresponding vector n.
    :return: Modified matrix and vector after row swapping.
    """

    n = len(matrix)

    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(matrix[r][i]))
        if i != max_row:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            vector[i], vector[max_row] = vector[max_row], vector[i]
    return matrix, vector


def GaussJordanElimination(matrix, vector):
    """
    Function for solving a linear equation using Gauss-Jordan elimination method
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=A(-1)b
    """
    # Pivoting process
    matrix, vector = RowXchange(matrix, vector)
    # Inverse matrix calculation
    #invert = InverseMatrix(matrix, vector)
    invert = np.linalg.inv(matrix)
    return MulMatrixVector(invert, vector)


def UMatrix(matrix,vector):
    """
    :param matrix: Matrix nxn
    :return:Disassembly into a U matrix
    """
    # result matrix initialized as singularity matrix
    U = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i])/matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    # U matrix is a doubling of elementary matrices that we used to reset organs under the pivot
    U = MultiplyMatrix(U, matrix)
    return U


def LMatrix(matrix, vector):
    """
       :param matrix: Matrix nxn
       :return:Disassembly into a  L matrix
       """
    # Initialize the result matrix
    L = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i])/matrix[i][i]
            # L matrix is a doubling of inverse elementary matrices
            L[j][i] = (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)

    return L


def SolveLU(matrix, vector):
    """
    Function for deconstructing a linear equation by ungrouping LU
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=U(-1)L(-1)b
    """
    matrixU = UMatrix(matrix)
    matrixL = LMatrix(matrix)
    return MultiplyMatrix(InverseMatrix(matrixU), MultiplyMatrix(InverseMatrix(matrixL), vector))

def solveMatrix(matrixA,vectorb):
    """
    Solve a system of linear equations using different methods based on matrix properties.

    :param matrixA: Coefficient matrix.
    :param vectorb: Result vector.
    :return: Solution vector.
    """
    detA = Determinant(matrixA, 1)
    print(bcolors.GOLD, "\nDET(A) = ", detA)
    if detA != 0:
        print("CondA = ", Cond(matrixA, InverseMatrix(matrixA, vectorb)), bcolors.ENDC)
        print(bcolors.OKBLUE, "\nnon-Singular Matrix - Perform GaussJordanElimination",bcolors.ENDC)
        result = GaussJordanElimination(matrixA, vectorb)
        print(np.array(result))
        return result
    else:
        print("Singular Matrix - Perform LU Decomposition\n")
        print("Matrix U: \n")
        print(np.array(UMatrix(matrixA, vectorb)))
        print("\nMatrix L: \n")
        print(np.array(LMatrix(matrixA, vectorb)))
        print("\nMatrix A=LU: \n")
        result = MultiplyMatrix(LMatrix(matrixA, vectorb), UMatrix(matrixA, vectorb))
        print(np.array(result))
        return result

def linear_interpolation(xList, yList, point):
    """
    Performs linear interpolation for a given set of data points.

    Parameters:
        xList (list): List of x-values of the data points (must be sorted in ascending order).
        yList (list): List of y-values of the data points corresponding to xList.
        point (float): The x-value at which to evaluate the interpolation.

    Returns:
        float: Interpolated y-value at the given x-value.
        None: If the point is outside the range of the data.

    Notes:
        - The function assumes that `xList` is sorted in ascending order. If it is not,
          the results may be incorrect or the function could fail.
        - If the `point` lies exactly on one of the x-values in `xList`, the corresponding
          y-value from `yList` is returned directly.
        - The function will return `None` and print an error message if the `point` is
          outside the range `[min(xList), max(xList)]`.
    """
    result = 0
    if x in xList:
        print(bcolors.OKGREEN, f"\nPoint is in the data", bcolors.ENDC)
        return yList[xList.index(x)]

    for i in range(len(xList) - 1):
        if xList[i] <= point <= xList[i + 1]:
            x1, x2 = xList[i], xList[i + 1]
            y1, y2 = yList[i], yList[i + 1]
            result = (((y1 - y2) / (x1 - x2)) * point) + (((y2 * x1) - (y1 * x2)) / (x1 - x2))
    if result != 0:
        return result
    else:
        print (bcolors.FAIL, f"\nError: x = {x} is outside the range of the given data points.", bcolors.ENDC)
        return None

def polynomialInterpolation(xList, yList, x):
    """
    Performs polynomial interpolation and extrapolation using the Gauss-Seidel method.

    Parameters:
        xList (list): List of x-values (size n).
        yList (list): List of y-values corresponding to xList (size n).
        x (float): The x-value for which p(x) is to be calculated.

    Returns:
        float: Interpolated value at the given x-value.
        None: If x is outside the interpolation range or if the system does not converge.
    Notes:
        - The function constructs a system of linear equations to solve for the coefficients
          of the polynomial using the Gauss-Jordan Elimination method (or LU decomposition).
        - The function returns None if the x-value is outside the interpolation range.
    """
    n = len(xList)

    # Step 1: Check if x is in xList
    if x in xList:
        print(bcolors.OKGREEN ,f"\nPoint is in the data, p({x}) = {yList[xList.index(x)]}", bcolors.ENDC)
        return yList[xList.index(x)]

    # Step 2: Check if x is within range
    if x < min(xList) or x > max(xList):
        print(bcolors.WARNING ,f"\nError: x = {x} is outside the interpolation range [{min(xList)}, {max(xList)}].", bcolors.ENDC)
        return None

    # Step 3: Construct the matrix
    A = [[xi**j for j in range(n)] for xi in xList]

    # Step 4: Initialize the solution vector b
    b = yList[:]

    # print the matrix A and vector b
    print(bcolors.OKBLUE, "\nMatrix from the points: \n", bcolors.ENDC)
    print(np.array(A))
    print(bcolors.OKBLUE ,"\nVector b:", bcolors.ENDC, b, "\n")

    # Step 5: Solve for coefficients using Gauss-Jordan Elimination or LU Decomposition
    matrixSol = solveMatrix(A, b)

    # Step 6: Compute result using the polynomial
    result = sum([matrixSol[i] * (x ** i) for i in range(n)])
    print(f"{bcolors.OKBLUE}\nThe polynomial:{bcolors.ENDC}")
    print(f"P(X) = {' + '.join([f'({matrixSol[i]}) * x^{i}' for i in range(len(matrixSol))])}")

    # Step 7: return the result
    return result


#main
xList = [0, 1, 2, 3]  # Known x-values
yList = [0, 0.8415, 0.9093, 0.1411]  # Corresponding y-values
x = 2.5 # Point to interpolate
print(bcolors.OKBLUE, "==================== Linear / Polynomial Interpolation Methods ====================\n", bcolors.ENDC)
while True:
    print("Please choose the method you want to use:")
    print("1. Linear Method")
    print("2. Polynomial Method")
    try:
        choice = int(input("Enter your choice: "))
        if choice in [1, 2]:
            break
        else:
            print(bcolors.WARNING,"Invalid choice. Please enter 1 or 2.", bcolors.ENDC)
    except ValueError:
        print(bcolors.WARNING,"Invalid input. Please enter a number (1 or 2).", bcolors.ENDC)
print("================================================================================")

if choice == 1:
    print(bcolors.OKBLUE, "You have chosen the Linear Method.", bcolors.ENDC)
    result = linear_interpolation(xList, yList, x)
    if result is not None:
        print(bcolors.OKGREEN, f"\nThe approximate (interpolation) of the point p({x}) = {result}", bcolors.ENDC)
else:
    print(bcolors.OKBLUE, "You have chosen the Polynomial Method.", bcolors.ENDC)
    result = polynomialInterpolation(xList, yList, x)
    if result is not None:
        print(bcolors.OKGREEN, f"\nThe approximate (interpolation) of the point P({x}) = {result}", bcolors.ENDC)

print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n",bcolors.ENDC)