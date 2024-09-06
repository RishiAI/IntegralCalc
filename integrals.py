import numpy as np
import math 

'''
hi im hrishi and this is my integral calculator. its not necessarily the most accurate; im sure you can find better on like symbolab or something
i wrote this as a personal project to get to understand non-Riemann integrals a little better. for each of these integrals i will try to explain my steps,
but as a high schooler who hasn't taken real analysis, measure theory, stochastic processes etc. you will have to bear with me as there will probably be some inaccuracies. 
feel free to play with the code and test your own functions. this project is currently incomplete
'''
#these are our test functions
def f(x):
    return x**2
def g(x): 
    return math.sin(x)
def h(x):
    return x
def dirichlet(x):
    #returns 1 if x is rational, 0 if it isnt. this is specifically to test the lebesgue integral. should return 0
    return 1 if np.isclose(x, round(x), atol=1e-10) else 0
def cauchy_pdf(x):
    #probability density function of the cauchy distribution. also used to test lebesgue integral. should return approx 1.
    return 1 / (np.pi * (1 + x**2))
def fourier_series(x, n_terms=10):
   #approximation of a square wave. also used to test lebesgue integral. should return 1
    result = 0
    for n in range(1, n_terms + 1, 2):
        result += (4 / (np.pi * n)) * np.sin(n * np.pi * x)
    return result


def riemann(f,a,b,n): #     this is our simple riemann. you should know how to do this one. if not, the code should explain. f is our function, a is our left limit and b is our right. n is the amt of rectangles that we're summing. the more we use, the more accurate this measurement is. 
    dx = (b-a)/n #    defining our infinitesimal change in x (its not actually infinitesimal but i would hope that you plug in an n value large enough for it to be accurate)
    totalleft = 0
    totalright = 0#   initializing our sums
    for i in range(n): #breaking down the function into n rectangles with width dx and height of the function at that point.
        totalleft += (dx)* f(a+(dx)*i) #sum up the rectangles from left endpoint to right
        totalright += (dx)* f(b-(dx)*i)#sum up the rectangles from right endpoint to left
    return (totalleft + totalright)/2 #average for accuracy! over and under estimates get cancelled out. now its comparable to the darboux in terms of accuracy, in some cases better.

print (riemann(f,0,1,1000)) #test riemann integral for f(x) = x^2 from 0-1
print (riemann(g,0,3*math.pi/2,1000)) #test riemann integral for f(x) = sin(x) from 0 to 3pi/2


def darboux(f,a,b,n): #    this is not so different from a riemann. however, instead of using our left endpoint and/or right endpoint as our height values, we take the supremum or infimum of our infinitely small rectangles
    x = np.linspace(a, b, n+1, dtype=np.float64) #      dont worry about this, just saving x as a vector so that i can store multiple scalar values in for my xi's
    dx = (b-a)/n  
    y = np.array([f(xi) for xi in x], dtype=np.float64) #       creatng an array to store my n # of f(xi) values 
    
    lower_sum = np.sum(np.minimum(y[:-1], y[1:]) * dx) #        finding the min for each rectangle, its a lot of rectangles
    upper_sum = np.sum(np.maximum(y[:-1], y[1:]) * dx) #        finding the max
    
    return (lower_sum + upper_sum)/2  #averaging it out for accuracy! over and underestimates get balanced out. ofc, if your n value is high enough this should not make much of a difference

print (darboux(f,0,1,1000)) #test darboux integral for f(x) = x^2 from 0-1
print (darboux(g,0,3*math.pi/2,1000)) #test darboux integral for f(x) = sinx from 0-3pi/2


def stieltjes(f, g, a, b, n): #    the stieltjes integral is different than the previous two. now we're taking an integral with respect to alpha(x). for the sake of not adding in annoying symbols we'll call alpha g. 
    dx = (b-a) / n
    total = 0
    for i in range(n):
        x = a + i * dx
        total += f(x) * (g(x + dx) - g(x))
    return total

print(stieltjes(f,h, 0,1,1000))
print(stieltjes(g,h,0,3*math.pi/2,1000)) #test stieljes integral for f(x) = sin(x) from 0-3pi/2

#--------------------------------------------------------------------------------------------------
#this is where it gets serious and the integrals get a lot lot lot harder

def measure_preimage(f, y_min, y_max, a, b, n): #this is a method that our lebesque will have to use
    
    #estimate the measure of the set of points in [a, b] that map to [y_min, y_max] under f.
    
    x_values = np.linspace(a, b, n)     # generating sample points in the domain
    f_values = np.array([f(x) for x in x_values]) # eval functions at said points
    preimage_count = np.sum((f_values >= y_min) & (f_values < y_max)) # count how many of the function values fall within [y_min, y_max]
    measure = (preimage_count / n) * (b - a) # estimate the measure as the proportion of points in the interval times the length of the interval
    
    return measure

def lebesgue(f, a, b, r, n):  #
    """
    parameters:
    - f: the function to integrate
    - a, b: the integration bounds
    - r: mumber of intervals to partition the range of f into
    - domain_samples: number of samples to take in the domain to approximate the measure
    """
    x_samples = np.linspace(a, b, n)
    f_samples = np.array([f(x) for x in x_samples]) # sample the function in the domain to get an idea of the range
    
    
    f_min, f_max = np.min(f_samples), np.max(f_samples)# determine the range of the function
    
    y_values = np.linspace(f_min, f_max, r+1) # partition the range into intervals
    total = 0.0
    for i in range(r):
        y_min, y_max = y_values[i], y_values[i+1]
        measure = measure_preimage(f, y_min, y_max, a, b, n)
        mid_value = (y_min + y_max) / 2 #take middle height for good guesstimate height of layer
        total += mid_value * measure 
    
    return total
print(lebesgue(dirichlet, 0, 1, 100, 10000)) 

"""
this is not 100% accurate. the answer should be 0. theoretically if our r & n value was infinity, this would be 0, but i highly recommend not plugging in bigger numbers than the ones above. 
1- this run itself takes a decent chunk of time - around 10-20 seconds to calculate the integral. this gets EXPONENTIALLY higher with each increase in r, and linearly with each increase in n. 
i tried plugging in 5000 as my r-value and had to let it run overnight for it to find the answer which came out with only 5 digit accuracy, compared to the 3 that it computes in 30 seconds. not worth it.
2- your PC will be fighting demons. with the code above, it already has to run the same function a million times (not an exaggeration, an actual number).
you do NOT want to melt your CPU by running it a billion times more for 2 extra digits of accuracy.

the same can be said for the next two which are both supposed to be 1, but will be a few decimals off.
"""
print(lebesgue(cauchy_pdf, -10, 10, r=100, n=10000))
print(lebesgue(lambda x: fourier_series(x, n_terms=50), 0, 1, r=100, n=10000))
