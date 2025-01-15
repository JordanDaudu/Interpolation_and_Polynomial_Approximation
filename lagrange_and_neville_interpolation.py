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

def lagrange_interpolation(xList, yList, x):
    """
    Evaluate the interpolated polynomial using Lagrange Interpolation.

    Parameters:
        xList (list): List of distinct x-values for the data points.
        yList (list): List of corresponding y-values for the data points.
        x (float): The x-value where the polynomial is to be evaluated.

    Returns:
        float: The interpolated y-value at x.
        None: If x is outside the interpolation range.
    Notes:
        - If x is one of the x-values in the data, the corresponding y-value is returned.
        - If x is outside the interpolation range, an error message is printed and None is returned.
    """
    if x in xList:
        print(bcolors.OKGREEN ,f"\nPoint is in the data", bcolors.ENDC)
        return yList[xList.index(x)]

    if x < min(xList) or x > max(xList):
        print(bcolors.FAIL ,f"\nError: x = {x} is outside the interpolation range [{min(xList)}, {max(xList)}].", bcolors.ENDC)
        return None

    n = len(xList)
    result = 0.0

    for i in range(n):
        term = yList[i]
        for j in range(n):
            if i != j:
                term *= (x - xList[j]) / (xList[i] - xList[j])
        result += term

    return result

def neville_interpolation(xList, yList, x):
    """
    Evaluate the interpolated polynomial using Neville's Method.

    Parameters:
        xList (list): List of distinct x-values for the data points.
        yList (list): List of corresponding y-values for the data points.
        x (float): The x-value where the polynomial is to be evaluated.

    Returns:
        float: The interpolated y-value at x.
        None: If x is outside the interpolation range.
    Notes:
        - If x is one of the x-values in the data, the corresponding y-value is returned.
        - If x is outside the interpolation range, an error message is printed and None is returned.
        - Intermediate polynomials are printed to show the step-by-step calculation.
    """
    if x in xList:
        print(bcolors.OKGREEN, f"\nPoint is in the data", bcolors.ENDC)
        return yList[xList.index(x)]

    if x < min(xList) or x > max(xList):
        print(bcolors.FAIL, f"\nError: x = {x} is outside the interpolation range [{min(xList)}, {max(xList)}].",
              bcolors.ENDC)
        return None

    n = len(xList)
    if not (xList[0] <= x <= xList[-1]):
        print("x is not in the interpolation range")
        return None

    P = [[0 for _ in range(n)] for _ in range(n)]

    # Initialize the diagonal with the y values
    for i in range(n):
        P[i][0] = yList[i]

    # Calculate the rest of the table
    for j in range(1, n):
        for i in range(n - j):
            P[i][j] = ((x - xList[i + j]) * P[i][j - 1] - (x - xList[i]) * P[i + 1][j - 1]) / (xList[i] - xList[i + j])

    # Print intermediate polynomials in the desired format
    print(bcolors.OKCYAN ,"\nIntermediate polynomials:", bcolors.ENDC)
    for i in range(n):
        print(bcolors.MAGENTA ,f"p{i} = {P[i][0]}", bcolors.ENDC)

    for j in range(1, n):
        for i in range(n - j):
            print(bcolors.MAGENTA ,f"p{i}{i + j} = {P[i][j]}", bcolors.ENDC)

    # final result
    return P[0][n - 1]

#main
xList = [1, 1.2, 1.3,1.4]
yList = [0, 0.112463, 0.167996, 0.222709]
x = 1.28

# xList = [1, 2, 4]
# yList = [1, 0, 1.5]
# x = 3
print(bcolors.OKBLUE, "==================== Lagrange / Neville Interpolation Methods ====================\n", bcolors.ENDC)
while True:
    print("Please choose the method you want to use:")
    print("1. Lagrange Method")
    print("2. Neville Method")
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
    print(bcolors.OKBLUE, "You have chosen the Lagrange Method.", bcolors.ENDC)
    result = lagrange_interpolation(xList, yList, x)
    if result is not None:
        print(bcolors.OKGREEN ,f"\nLagrange Interpolation: p(x = {x}) = {result}", bcolors.ENDC)
else:
    print(bcolors.OKBLUE, "You have chosen the Neville Method.", bcolors.ENDC)
    result = neville_interpolation(xList, yList, x)
    if result is not None:
        print(bcolors.OKGREEN ,f"\nNeville Interpolation: p(x = {x}) = {result}", bcolors.ENDC)