# Speech and language processing

Notes from reading the
[book of the same name](https://web.stanford.edu/~jurafsky/slp3/).

## Notes

### Chapter 2

Long interlude on regular expressions.

Book uses **case-sensitive** regular expressions.

> General principal: corpus may be parsed into syntax trees? There will be a
> collection of syntax tree types and appropriate mappings between those types.
> I.e., in one syntax tree type "Hi" and "hi" may be different, but there is the
> obvious "projection" mapping to the type where they are the same. These syntax
> trees must all project down to a sequence of characters.

They also use lookahead expressions.

The size of a vocabulary is &approx; k &times; N<sup>0.7</sup>, i.e a bit faster
than square root in terms of the number of tokens (**Herdan's law**).

:warning: In NLP, "type" means "distinct word in corpus", so that tokens map
to types.

#### 2.4 Text Normalization

// I see text normalization as mapping from finer to "coarser" synatx tree
// types.

Unix command prompt provides simple tools for summarizing a corpus.

Python has a commonly used NLP package &mdash; [NLTK](http://www.nltk.org).

Tokenization is very language-dependent.

"Byte pair encoding" (BPE) is a way of learning the dictionary directly from
character representation of a corpus.

BERT used a "wordpiece" tokenizer.

Normalization, case folding, lemmatization &mdash; all seem like "syntax tree
projections".

Minimum edit distance algorithms use dynamic programming.

### Chapter 3

**Language model** is a model which assigns probabilities to sequences of
words.

General framework: n-gram models.

Use log probabilities to avoid arithmetic overflows.

Use "unknown" of "out of vocabulary" (OOV) dummy words to handle data not in
the training corpus.

### Chapter 4

There are many text classification problems (e.g., good vs. bad review, which
language is a tweet written in).

This is a canonical problem of supervised machine learning.

Generative vs. discriminative classifiers.

Na&iuml;ve Bayes uses "bag of words" representation.

// This violates the "everything can project to sequence of chars" assumption.

Manual "hand crafted" algorithm for training this.

----

Last page: 64.
