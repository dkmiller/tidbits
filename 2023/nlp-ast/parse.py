import nltk
from nltk import pos_tag, word_tokenize, RegexpParser, Tree


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Extract all parts of speech from any text
chunker = RegexpParser(
    """
                       NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases
                       P: {<IN>}               #To extract Prepositions
                       V: {<V.*>}              #To extract Verbs
                       PP: {<p> <NP>}          #To extract Prepositional Phrases
                       VP: {<V> <NP|PP>*}      #To extract Verb Phrases
                       """
)

# Def "Discrimination"
text = """
The differential treatment of an individual or group of people based on their
race, color, national origin, religion, sex (including pregnancy and gender
identity), age, marital and parental status, disability, sexual orientation,
or genetic information.
"""

# Find all parts of speech in above sentence
tagged = pos_tag(word_tokenize(text))

# Print all parts of speech in above sentence
tree = chunker.parse(tagged)
print(f"After Extracting\n{tree}")

assert isinstance(tree, Tree)

tree.draw()
