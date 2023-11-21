# NLP syntax trees

[Syntax Tree &mdash; Natural Language Processing](https://www.geeksforgeeks.org/syntax-tree-natural-language-processing/)

[U.S. Department of the Interior &gt; What are Discrimination, Harassment, Harassing Conduct, and Retaliation?](https://www.doi.gov/employees/anti-harassment/definitions)

[`nltk.grammar` module](https://www.nltk.org/api/nltk.grammar.html)

[Context Free Grammars](http://aritter.github.io/courses/5525_slides/cfg.pdf)

"Tags" (e.g. `JJ`) come from the
[Brown corpus](https://varieng.helsinki.fi/CoRD/corpora/BROWN/tags.html)
([wiki](https://en.wikipedia.org/wiki/Brown_Corpus)).

[Brown corpus manual](http://korpus.uib.no/icame/manuals/BROWN/INDEX.HTM)

Relevant bits:

| Tag | Meaning | Example |
|-|-|-|
| `NN` | singular or mass noun | race, pregnancy |
| `VP` | verb phrase | gender identity |
| `CC` | coordinating conjunction | or |

https://cs.nyu.edu/~grishman/jet/guide/PennPOS.html

[Natural Language Processing Tag definitions](https://stackoverflow.com/q/21514270/)

Better:
[Java Stanford NLP: Part of Speech labels?](https://stackoverflow.com/q/1833252/)
&mapsto;
[Penn Treebank II Constituent Tags](http://surdeanu.cs.arizona.edu//mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html)

## Alternatives

- [English grammar for parsing in NLTK](https://stackoverflow.com/q/6115677/)
- [how to get parse tree using python nltk?](https://stackoverflow.com/q/42322902/)

----

> The differential treatment of an individual or group of people based on their
race, color, national origin, religion, sex (including pregnancy and gender
identity), age, marital and parental status, disability, sexual orientation,
or genetic information.

&mapsto;

> The differential treatment of an `${ind_or_group}` based on their `${identify}`

- `ind_or_group` = "individual" + "group of people"
- `identify` = "race" + "color" + "national origin" + "religion" + "sex" + "pregnancy" + "gender identity" + "age" + "marital and parental status" + "disability" + "sexual orientation" + "genetic information"

Look for sequences

something`,|or`something...

Hmm... this may not be easy from parse trees.

&mapsto;

> The differential treatment of an individual based on their being White

&mapsto;

> The differential treatment of `${name}` based on their being White

&mapsto;

> Treat Daniel differently based on their being White.
