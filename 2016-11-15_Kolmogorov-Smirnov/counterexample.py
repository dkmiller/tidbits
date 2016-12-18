import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

matplotlib.style.use('ggplot')

def U_1(theta):
    return np.sin(2*theta) / np.sin(theta)

def cdf_sin(x):
    return np.sin(x/2) ** 2

def cdf_ST(x):
    return (x - np.sin(x)*np.cos(x)) / np.pi

def primes_below(x):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    # Input x in R, returns an array of primes p <= x.
    if x < 2:
        return []
    elif x < 3:
        return [2]
    elif x < 5:
        return [2,3]
    elif x < 7:
        return [2,3,5]
    n = int(x) + 1
    sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in xrange(int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)/3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))/3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

def traces(p):
    bound = int(2*np.sqrt(p))
    return np.array(range(-bound,bound + 1))

def thetas(p):
    return np.arccos(traces(p) / (2*np.sqrt(p)))

def ks(arr, cdf):
    return scipy.stats.kstest(arr, cdf).statistic

def good_thetas(p, partial_sum):
    c = 2
    result = []
    for theta in thetas(p):
        if abs(partial_sum + U_1(theta)) <= c*np.sqrt(p):
            result.append(theta)
    return result

def find_traces(x):
    c = 4
    result = {}
    if x < 2:
        return result
    primes = primes_below(x)
    for p in primes:
        partial_sum = sum(map(U_1, result.values()))
        for theta in good_thetas(p, partial_sum):
            bound = 1 / np.log(p)
            if bound/c <= ks(result.values() + [theta], cdf_ST) <= c*bound:
                #print 'Found theta = %f for p = %d' % (theta, p)
                result[p] = theta
                break
        if p not in result:
            print 'No theta found for p = %d!' % p
            break
    return result

def test_traces(d):
    p_max = max(d.keys())
    if abs(sum(map(U_1, d.values()))) > 2*np.sqrt(p_max):
        print '|sum| is too big!'
    kol_smi = ks(d.values(), cdf_ST)
    if kol_smi < 1/(4*np.log(p_max)) or kol_smi > 4 / np.log(p_max):
        print 'Kolmogorov-Smirnov statistic is too large!'

# stuff = find_traces(x)
def display_ks(stuff):
    p_max = max(stuff.keys())
    ks_data = []
    temp = []
    for p in primes_below(p_max):
        temp.append(stuff[p])
        ks_data.append(ks(temp, cdf_ST))
    plt.show(plt.scatter(primes_below(p_max), ks_data))

def display_sum(stuff):
    p_max = max(stuff.keys())
    sum_data = []
    partial_sum = 0
    primes = primes_below(p_max)
    for p in primes:
        partial_sum += U_1(stuff[p])
        sum_data.append(partial_sum)
    plt.show(plt.scatter(primes, np.abs(sum_data)))
