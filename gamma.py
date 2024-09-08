import math
from integrals import riemann

def gamma_integrand(t, z):
    return t**(z-1) * math.exp(-t)

def gamma(z, n=1000):
    upper_limit = 100
    
    f = lambda t: gamma_integrand(t, z)
    
    result = riemann(f, 0, upper_limit, n)
    
    return result

def factorial_approximation(n):
    return gamma(n + 1)

def compare_factorial(n):
    approx = factorial_approximation(n)
    actual = math.factorial(n)
    relative_error = abs(approx - actual) / actual
    return approx, actual, relative_error

if __name__ == "__main__":

    test_values = [1, 2, 3, 4, 5]
    for z in test_values:
        gamma_result = gamma(z)
        expected = math.factorial(z-1)
        print(f"Gamma({z}):")
        print(f"  Approximated: {gamma_result}")
        print(f"  Actual: {expected}")
        print(f"  Relative Error: {abs(gamma_result - expected) / expected:.6f}")
        print()
    
    