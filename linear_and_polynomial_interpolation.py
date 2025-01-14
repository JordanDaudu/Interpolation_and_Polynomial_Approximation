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
    Prints the header for the iteration table, and checks if matrix is diagonal matrix.
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
    """Check if a matrix is diagonally dominant."""
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True

def make_diagonally_dominant(A):
    """
    Modifies the matrix A to make it diagonally dominant by swapping rows if necessary.

    Parameters:
        A: The coefficient matrix to be modified.

    Returns:
        The modified matrix that is diagonally dominant (if possible).
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

def gauss_seidel(A, b, X0=None, TOL=0.00001, N=200, verbose=True):
    """
    Performs Gauss-Seidel iterations to solve Ax = b.

    Parameters:
        A: Coefficient matrix (list of lists).
        b: Solution vector (list).
        X0: Initial guess for the solution. Defaults to a zero vector.
        TOL: Tolerance for convergence. Defaults to 0.00001.
        N: Maximum number of iterations. Defaults to 200.
        verbose: If True, prints iteration details.

    Returns:
        Approximate solution vector.
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
    Perform polynomial interpolation and extrapolation.
    Uses the Gauss-Seidel iterative method for solving the coefficient matrix.

    Parameters:
        xList: List of x values (size n).
        yList: List of y values corresponding to xList (size n).
        x: The x value for which p(x) is to be calculated.

    Returns:
        Prints p(x).
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