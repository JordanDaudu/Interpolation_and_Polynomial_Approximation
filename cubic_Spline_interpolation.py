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

def print_iteration_header(A, b, verbose=True):
    """
    Prints the header for the iteration table and checks if the matrix is diagonally dominant.

    Parameters:
        A (list of lists): Coefficient matrix.
        b (list): Solution vector.
        verbose (bool): If True, prints additional diagnostic information.
    """
    n = len(A)

    if is_diagonally_dominant(A) and verbose:
        print(bcolors.ORANGE ,"Matrix is diagonally dominant.", bcolors.ENDC)
    if not is_diagonally_dominant(A):
        print(bcolors.ORANGE ,"Matrix is not diagonally dominant. Attempting to modify the matrix...", bcolors.ENDC)
        A, b = make_diagonally_dominant(A, b)
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

def make_diagonally_dominant(A, b):
    """
    Modifies the matrix A to make it diagonally dominant by swapping rows if necessary.
    Also swaps vector b accordingly.

    Parameters:
        A (list of lists): The coefficient matrix to be modified.
        b (list): The vector corresponding to the right-hand side of the equations.

    Returns:
        tuple: The modified matrix A and the modified vector b that are diagonally dominant (if possible).
    """
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
            # Find a row with a larger diagonal element
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[i][i]):
                    # Swap row i and row j in both matrix A and vector b
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
            # After attempting to swap, if no dominant diagonal is found, print a warning
            if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
                print(f"Warning: Row {i} still not diagonally dominant after attempting row swaps.")

    return A, b

def gauss_seidel(A, b, X0=None, TOL=0.00001, N=200, verbose=True):
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

    print_iteration_header(A, b, verbose)

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

    print(bcolors.WARNING, "Maximum number of iterations exceeded, Matrix is not converging", bcolors.ENDC)
    raise ConvergenceError("Gauss-Seidel method failed to converge within the maximum number of iterations.")



def cubic_spline(xList, yList, x, f_tag0, f_tagN):
    """
    Computes the cubic spline interpolation for a given set of data points.

    The function calculates both natural and full cubic spline interpolations
    at a given point `x`. Natural spline interpolation assumes the second
    derivative at the endpoints is zero, while the full spline uses provided
    derivative values (`f_tag0` and `f_tagN`) at the endpoints.

    Parameters:
        xList (list): List of x-coordinates of the data points.
        yList (list): List of y-coordinates corresponding to `xList`.
        x (float): The x-coordinate where the spline is evaluated.
        f_tag0 (float): First derivative of the function at the start of the interval.
        f_tagN (float): First derivative of the function at the end of the interval.

    Returns:
        tuple:
            - s_natural (float): The interpolated value using the natural spline.
            - s_full (float or None): The interpolated value using the full spline
              (or None if `f_tag0` and `f_tagN` are not provided).

    Notes:
        - If `x` is outside the range of `xList`, an error message is displayed,
          and the function returns `(None, None)`.
        - The Gauss-Seidel method is used to solve the system of linear equations
          for spline coefficients. If this method fails to converge, appropriate
          warnings are shown.
    """
    if x in xList:
        print(bcolors.OKGREEN ,f"\nPoint is in the data", bcolors.ENDC)
        print(bcolors.OKGREEN,f"x = {x}: y = {yList[xList.index(x)]}", bcolors.ENDC)
        return None, None

    # Remove this block if you want to allow extrapolation
    if x < min(xList) or x > max(xList):
        print(bcolors.FAIL ,f"\nError: x = {x} is outside the interpolation range [{min(xList)}, {max(xList)}].", bcolors.ENDC)
        return None, None

    n = len(xList)

    # Step 1: Calculate h_list
    h_list = [xList[i + 1] - xList[i] for i in range(n - 1)]

    # Step 2: Calculate lambda and mu lists
    lambda_list = [h_list[i] / (h_list[i - 1] + h_list[i]) for i in range(1, n - 1)]
    mu_list = [1 - lam for lam in lambda_list]

    # Step 3: Calculate d_list
    d_list = [6 * ((yList[i + 1] - yList[i]) / h_list[i] - (yList[i] - yList[i - 1]) / h_list[i - 1]) / (h_list[i - 1] + h_list[i])
              for i in range(1, n - 1)]

    # Step 4: Create the matrix and solve for natural spline cubic
    A = [[0] * n for _ in range(n)]  # Create an n x n zero matrix
    b = [0] * n  # Create an n x 1 zero vector

    # Fill the matrix for the natural spline
    A[0][0], A[-1][-1] = 1, 1  # Natural spline boundary conditions
    for i in range(1, n - 1):
        A[i][i - 1] = mu_list[i - 1]
        A[i][i] = 2
        A[i][i + 1] = lambda_list[i - 1]
        b[i] = d_list[i - 1]

    # Solve the linear system for M (natural spline)
    try:
        print(bcolors.OKCYAN ,"Solving for the natural spline...", bcolors.ENDC)
        M_natural = gauss_seidel(A, b, verbose=True)
    except ConvergenceError as e:
        print("Gauss-Seidel failed to converge for the natural spline system:", e)
        return None, None

    # Step 5: Calculate the natural spline result
    i = next(i for i in range(len(xList) - 1) if xList[i] <= x < xList[i + 1])
    h_i = h_list[i]
    s_natural = ((xList[i + 1] - x) ** 3 * M_natural[i] + (x - xList[i]) ** 3 * M_natural[i + 1]) / (6 * h_i)
    s_natural += ((xList[i + 1] - x) * yList[i] + (x - xList[i]) * yList[i + 1]) / h_i
    s_natural -= ((xList[i + 1] - x) * M_natural[i] + (x - xList[i]) * M_natural[i + 1]) * h_i / 6

    # Step 6: Solve for the full spline cubic
    # Adjust boundary conditions for full spline
    if f_tag0 is not None and f_tagN is not None:
        b[0] = 6 * ((yList[1] - yList[0]) / h_list[0] - f_tag0) / h_list[0]
        b[-1] = 6 * (f_tagN - (yList[-1] - yList[-2]) / h_list[0]) / h_list[-1]
        A[0][0], A[0][1], A[-1][-2], A[-1][-1] = 2, 1, 1, 2

        try:
            print(bcolors.OKCYAN, "\n Solving for the full spline...", bcolors.ENDC)
            M_full = gauss_seidel(A, b, verbose=True)
        except ConvergenceError as e:
            print("Gauss-Seidel failed to converge for the full spline system:", e)
            return s_natural, None
    else:
        print("Error: f_tag0 and f_tagN must be provided for full spline")
        return s_natural, None

    # Step 7: Calculate the full spline result
    if M_full is not None:
        s_full = ((xList[i + 1] - x) ** 3 * M_full[i] + (x - xList[i]) ** 3 * M_full[i + 1]) / (6 * h_i)
        s_full += ((xList[i + 1] - x) * yList[i] + (x - xList[i]) * yList[i + 1]) / h_i
        s_full -= ((xList[i + 1] - x) * M_full[i] + (x - xList[i]) * M_full[i + 1]) * h_i / 6
    else:
        s_full = None

    # Return results
    return s_natural, s_full


#main
x_list = [0, PI/6, PI/4, PI/2]  # Known x-values
y_list = [0, 0.5, 0.7072, 1]  # Corresponding y-values
#x_list = [1, 22, 4]  # Known x-values
#y_list = [1, 0, 1.5]  # Corresponding y-values
x = PI/3                   # The x-value to interpolate
#x = 2.5
f_tag0 = 1                # First derivative at the start
f_tagN = 0              # First derivative at the end

print(bcolors.OKBLUE, "==================== Cubic Spline Interpolation/Extrapolation ====================\n", bcolors.ENDC)
# Call the cubic_spline function
s_natural, s_full = cubic_spline(x_list, y_list, x, f_tag0, f_tagN)

# Display the results
if s_natural and s_full is not None:
    print(bcolors.OKBLUE, f"\nNatural Spline result at x = {x}:", bcolors.OKGREEN, f"y = {s_natural}", bcolors.ENDC)
    print(bcolors.OKBLUE ,f"\nFull Spline result at x = {x}:", bcolors.OKGREEN, f"y = {s_full}", bcolors.ENDC)
