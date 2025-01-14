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

class ConvergenceError(Exception):
    """Exception for handling non-convergence in iterative methods."""
    pass

def norm(vector):
    """Computes the infinity norm of a vector."""
    return max(abs(v) for v in vector)

def print_iteration_header(A, verbose=True):
    """
    Prints the header for the iteration table and checks if the matrix is diagonally dominant.

    Parameters:
        A (list of lists): Coefficient matrix.
        verbose (bool): If True, prints additional diagnostic information.
    """
    n = len(A)

    if is_diagonally_dominant(A) and verbose:
        print(bcolors.ORANGE ,"Matrix is diagonally dominant.", bcolors.ENDC)
    if not is_diagonally_dominant(A):
        print(bcolors.ORANGE ,"Matrix is not diagonally dominant. Attempting to modify the matrix...", bcolors.ENDC)
        A = make_diagonally_dominant(A)
        if is_diagonally_dominant(A) and verbose:
            print(bcolors.ORANGE ,"Matrix modified to be diagonally dominant:\n", A, bcolors.ENDC)

    if verbose:
        print("Iteration" + "\t\t\t".join([" {:>12}".format(f"x{i + 1}") for i in range(n)]))
        print("--------------------------------------------------------------------------------")

def is_diagonally_dominant(A):
    """
    Checks if a matrix is diagonally dominant.

    Parameters:
        A (list of lists): The matrix to check.

    Returns:
        bool: True if the matrix is diagonally dominant, False otherwise.
    """
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True

def make_diagonally_dominant(A):
    """
    Modifies the matrix A to make it diagonally dominant by swapping rows if necessary.

    Parameters:
        A (list of lists): The coefficient matrix to be modified.

    Returns:
        list of lists: The modified matrix that is diagonally dominant (if possible).
    """
    n = len(A)

    for i in range(n):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
            # Find a row with a larger diagonal element
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[i][i]):
                    # Swap row i and row j
                    A[i], A[j] = A[j], A[i]
                    break
            # After attempting to swap, if no dominant diagonal is found, print a warning
            if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
                print(bcolors.WARNING ,f"Warning: Row {i} still not diagonally dominant after attempting row swaps.", bcolors.ENDC)

    return A

def gauss_seidel(A, b, X0=None, TOL=0.00001, N=100, verbose=True):
    """
    Performs Gauss-Seidel iterations to solve the system of linear equations Ax = b.

    Parameters:
        A (list of lists): Coefficient matrix of size n x n.
        b (list): Solution vector size n.
        X0 (list, optional): Initial guess for the solution. Defaults to a zero vector.
        TOL (float, optional): Tolerance for convergence. Defaults to 0.00001.
        N (int, optional): Maximum number of iterations. Defaults to 200.
        verbose (bool, optional): If True, prints iteration details. Defaults to True.

    Returns:
        list: Approximate solution vector.

    Raises:
        ConvergenceError: If the method fails to converge within the maximum number of iterations.

    Notes:
        - The Gauss-Seidel method updates each component of the solution vector
          immediately after it is computed.
        - The convergence of the method is guaranteed if the coefficient matrix `A` is
          strictly diagonally dominant.
        - The norm of the difference between successive approximations is used
          as the convergence criterion.
        - If the matrix `A` is not diagonally dominant, the method may still converge
          in some cases, but this is not guaranteed. A warning will be displayed if
          convergence occurs despite the lack of diagonal dominance.
    """
    n = len(A)
    if X0 is None:
        X0 = [0.0] * n

    print_iteration_header(A, verbose)

    for k in range(1, N + 1):
        x = X0.copy()
        for i in range(n):
            sigma1 = sum(A[i][j] * x[j] for j in range(i))
            sigma2 = sum(A[i][j] * X0[j] for j in range(i + 1, n))
            x[i] = (b[i] - sigma1 - sigma2) / A[i][i]

        if verbose:
            print(f"{k:<15}" + "\t\t".join(f"{val:<15.10f}" for val in x))

        if norm([x[i] - X0[i] for i in range(n)]) < TOL:
            if not is_diagonally_dominant(A):
                print(bcolors.OKCYAN ,"\n|Warning: Matrix is not diagonally dominant, but the solution is within tolerance and converged.|", bcolors.ENDC)
            return x

        X0 = x.copy()

    print(bcolors.WARNING ,"Maximum number of iterations exceeded, Matrix is not converging", bcolors.ENDC)
    raise ConvergenceError("Gauss-Seidel method failed to converge within the maximum number of iterations.")

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
            result = (((y1 - y2) / (x1 - x2)) * point) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
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
          of the polynomial using the Gauss-Seidel method.
        - The function returns None if the x-value is outside the interpolation range.
        - If the Gauss-Seidel method fails to converge, a ConvergenceError is raised.
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

    # Step 5: Solve for coefficients using Gauss-Seidel
    try:
        coefficients = gauss_seidel(A, b)
    except ConvergenceError as e:
        print(bcolors.FAIL ,"Error in solving system:", e, bcolors.ENDC)
        return None

    # Step 6: Compute p(x) using the polynomial
    px = sum(coefficients[i] * (x**i) for i in range(n))

    # Step 7: return the result
    return px

#main
xList = [1, 2, 3]
yList = [0.8415, 0.9093, 0.1411]
x = 2.5
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
        print(bcolors.OKGREEN, f"\nThe approximate (interpolation) of the point p({x}) = {result}", bcolors.ENDC)

print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n",bcolors.ENDC)