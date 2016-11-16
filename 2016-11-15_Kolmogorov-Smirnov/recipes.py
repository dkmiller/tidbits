import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

def from_lpdata(name):
    filename = '%s_lpdata.txt' % name
    df = pd.from_csv(filename)

def histograms(df, heads):
    for h in heads:
        plt.clf()
	plt.hist(df.head(h).K, bins=200, normed=True)
	plt.xlabel(str(h))
	plt.savefig('%d.png' % h)

