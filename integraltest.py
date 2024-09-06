import integrals
import sympy as sp
import random

def determine_integral_type(func_str):
    x = sp.Symbol('x')
    try:
        func = sp.sympify(func_str)
    except sp.SympifyError:
        return "Unable to parse the function. Please check the syntax."

    # check for continuity
    try:
        continuous_domain = sp.calculus.util.continuous_domain(func, x, sp.S.Reals)
        is_continuous = continuous_domain == sp.S.Reals
    except NotImplementedError:
        # handle functions where symPy can't determine continuity
        is_continuous = False

    # check for differentiability
    def is_differentiable(func, x):
        try:
            deriv = sp.diff(func, x)
            if deriv.has(sp.zoo, sp.oo, -sp.oo):
                return False
            return True
        except sp.NotImplementedError:
            return False

    is_diff = is_differentiable(func, x)
    
    # check for bounded variation
    def has_bounded_variation(func, x):
        try:
            deriv = sp.diff(func, x)
            integral = sp.integrate(sp.Abs(deriv), (x, -sp.oo, sp.oo))
            return integral.is_finite
        except (sp.IntegrationError, sp.NotImplementedError):
            return False

    is_bounded_variation = has_bounded_variation(func, x)
    
    # check for stochastic processes
    is_stochastic = 'W' in func_str or 'dW' in func_str
    
    if is_stochastic:
        return "Itô Integral (for stochastic processes)"
    elif is_continuous and is_diff:
        return "Riemann Integral (for continuous, differentiable functions)"
    elif is_continuous:
        return "Darboux Integral (for continuous functions, possibly not differentiable)"
    elif is_bounded_variation:
        return "Stieltjes Integral (for functions of bounded variation)"
    else:
        return "Lebesgue Integral (for more general functions, including some discontinuous functions)"

# list of functions to test
functions_to_test = [
    "x**2",
    "1/x",
    "sin(x)",
    "abs(x)",
    "floor(x)",
    "exp(-x**2/2) / sqrt(2*pi)",  
    "W(t)", 
]

# run the determination function on each test function
for func in functions_to_test:
    result = determine_integral_type(func)
    print(f"Function: {func}")
    print(f"Recommended Integral Type: {result}")
    print()

# user input
user_function = input("Enter a function: ")
user_result = determine_integral_type(user_function)
print(f"Recommended Integral Type: {user_result}")