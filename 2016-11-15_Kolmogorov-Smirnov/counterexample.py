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

def test_choose_traces(x):
    def theta(a, p):
        return np.arccos(a / (2*np.sqrt(p)))

#    primes = primes_below(x)
    # Ensure an even number of primes.
#    if len(primes) % 2 != 0:
#        primes = primes[:-1]
    primes_even = primes_below(x)[::2]
    primes_odd = primes_below(x)[1::2]

    # Drop primes if necessary so len(primes_odd) = len(primes_even).
    if len(primes_odd) > len(primes_even):
        primes_odd = primes_odd[:-1]
    elif len(primes_even) > len(primes_odd):
        primes_even = primes_even[:-1]
    traces_odd = map(lambda p: max(traces(p)), primes_odd)
    traces_even = []
    for i in xrange(len(primes_odd)):
        p = primes_odd[i]
        a_p = traces_odd[i]
        q = primes_even[i]
        def evaluate(a):
            return np.abs(theta(a_p,p) - (np.pi - theta(a,q)))
        a_q = min(traces(q))
        for a in traces(q):
            if evaluate(a) < evaluate(a_q):
                a_q = a
        traces_even.append(a_q)

    thetas_odd = []
    thetas_even = []
    for i in xrange(len(primes_odd)):
        thetas_odd.append(theta(traces_odd[i], primes_odd[i]))
        thetas_even.append(theta(traces_even[i], primes_even[i]))
    result = thetas_odd + thetas_even
    result[::2] = thetas_odd
    result[1::2] = thetas_even
    return result
