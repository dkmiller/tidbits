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


# Sample algorithm for choosing a_p's to minimize |sum U_1(theta_p)|.
# This algorithm yields |sum_{p<x} U_1(theta_p)| = O(1). 
def test_choose_traces(x):
    # Returns the angle corresponding to a = tr(fr_p).
    def theta(a, p):
        return np.arccos(a / (2*np.sqrt(p)))

    # Relevant set of primes; ensure there is an even number.
    primes = primes_below(x)
    if len(primes) % 2 != 0:
        primes = primes[:-1]

    local_traces = []
    local_thetas = []

    # Loop through pairs p < q of successive primes.
    for i in xrange(len(primes) / 2):
        p = primes[2*i]
        q = primes[2*i+1]

        # Choose a_p and theta_p.
        a_p = max(traces(p))
        if 2*np.sqrt(p) - a_p > 0.5:
            a_p = -a_p
        local_traces.append(a_p)
        theta_p = theta(a_p, p)
        local_thetas.append(theta_p)

        # Choose a_q and theta_q.

        # Returns |theta_p - (pi - theta_q)|.
        def evaluate(a):
            return np.abs(theta_p - (np.pi - theta(a, q)))
        a_q = min(traces(q))
        for a in traces(q):
            if evaluate(a) < evaluate(a_q):
                a_q = a
        local_traces.append(a_q)
        theta_q = theta(a_q, q)
        local_thetas.append(theta_q)

    return local_thetas
