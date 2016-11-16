import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats.kstest as kstest

def from_lpdata(name):
    filename = '%s_lpdata.txt' % name
    df = pd.from_csv(filename)

for i in xrange(1,67):
    num = 10000*i
    plt.clf()
    plt.hist(df.head(num).K, bins=200, normed=True)
    plt.xlabel('First %d primes' % num)
    plt.savefig('%d.png' % i)
