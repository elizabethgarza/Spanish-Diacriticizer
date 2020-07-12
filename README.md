# SpanishDiacriticizer | Supervised machine learning using the Naive Bayes classification algorithm.

## Purpose 

The purpose of this project is to design a supervised machine learning Naive Bayes classification algorithm that can predict whether or not a letter in a Spanish word should have a diacritic.  Aside from designing a `SpanishDiacriticizer`--i.e. a model that correctly adds diacritics to unidecoded Spanish words in a sentence--the other purpose of this project is to explore the pros and cons of using both the NLTK and Scikit versions of this algorithm.  

As such, you will see two directories, titled `nltk` and `sk-learn`, which contain python scripts that will give you the option to: 
1.  diacriticize Spanish texts based on the two models that have been made available on this repo.
2.  use your own Spanish data to train and evaluate your own model.

## Usage


### Background 

For some unidecoded words, like "espanol", the model simply has to perform a dictionary lookup to determine whether or not the "n" should have a diacritic. But for others, like "esta", this operation will not suffice because there are two viable word forms stored in the value entry: "está" and "esta". In such cases, the model has to be given more information about the words before it can predict whether or not to add a diacritic to the letter "a".  

Unlike computers, humans would quickly be able to decide which of the two words is correct by analyzing the word’s grammatical context.  For example, in this sentence--

`Esa casa [esta] sucia. / That house [is] dirty.`

--a writer can quickly determine that the correct form of the word "esta" should be "está" because it functions as the verb, "is," instead of the adjective "this". 

Outside of a grammatical context, there are also three simple pronunciation-based rules which humans can rely on, as long as the word is not a homophone: 

1. Words ending in a vowel, -n, or -s are stressed on the next to the last (penultimate) syllable--e.g.: 
   *na-da* is pronounced **na**-*da*.

2. Words ending in any consonant except -n or -s are stressed on the last syllable.--e.g. 
   *doc-tor* is pronounced *doc*–**tor**

3. When rules #1 and #2 above are not followed, a written accent is used--e.g.
   *com-pró* is pronounced *com*–**pró**; 
   *álbum* is pronounced *ál*–**bum**.

For a non-homophonic unidecoded word like *calculo*, which can appear in three different forms {*calculo*, *cálculo*, *calculó*}, humans could also think about how the word is pronounced to determine the correct spelling.

Because the computer will only be able to interpret written text, the pronunciation-based rules described above will not be helpful for this task.  (Even if it could, using these rules in isolation would not be able to disambiguate frequently used homophones like {*si*, *sí*}, {*cómo*, *como*}, {*sólo*, *solo*}, etc.)  And while one could write an algorithm based on grammatical rules, such a system would quickly become too complex to be feasibly executed. One popular alternative which I've decided to use for this project is the Naive Bayes classification algorithm, which can learn to make predictions without explicit instruction. Aside from designing a model that correctly adds diacritics to unidecoded words in a sentence, the other purpose of this project is to explore the pros and cons of using both the NLTK and Scikit multinomial versions of this algorithm.

### Description of algorithm 

### Data collection

### Methodology

### Results and discussion 

