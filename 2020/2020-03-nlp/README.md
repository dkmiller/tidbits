# Natural language processing

Based on the class
[CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/),
taught at Stanford.

## Setup

This code uses [Conda](https://docs.conda.io/en/latest/) for dependency
management. If you already have Conda installed, run the following code to
install the necessary dependencies and activate your environment.

```powershell
# Create environment.
conda env create --file environment.yml

# Activate environment.
conda activate nlp

# Run unit tests.
pytest
```

Optionally, after running all the code you may call
`conda remove --name nlp --all` to remove the environment.

## Links

- [Accessing Text Corpora and Lexical Resources](https://www.nltk.org/book/ch02.html)
- [Type hints cheat sheet (Python 3)](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [[feature request] [pytorch] Truncated SVD](https://github.com/pytorch/pytorch/issues/8049)
- [Pandas DataFrame column to list [duplicate]](https://stackoverflow.com/a/23749057)
- [How to update an existing Conda environment with a `.yml` file](https://stackoverflow.com/a/43873901)
- [Introduction to Matplotlib in Python](https://towardsdatascience.com/introduction-to-matplotlib-in-python-5f5a9919991f)
- [The Lifecycle of a Plot](https://matplotlib.org/3.1.1/tutorials/introductory/lifecycle.html)
- [Easily add Anaconda Prompt to Windows Terminal to make life better](https://dev.to/azure/easily-add-anaconda-prompt-in-windows-terminal-to-make-life-better-3p6j)
- [view and then close the figure automatically in matplotlib?](https://stackoverflow.com/a/40395799)
- [downloader &mdash; Downloader API for gensim](https://radimrehurek.com/gensim/downloader.html)
- [Skip and xfail: dealing with tests that cannot succeed](http://doc.pytest.org/en/latest/skipping.html)
