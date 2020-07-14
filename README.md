# nltk-Spanish-diacriticizer | Supervised machine learning using the Naive Bayes classification algorithm.

## Purpose 

The purpose of this project is to design model based on a supervised machine learning Naive Bayes classification algorithm that can predict whether or not a letter in a Spanish word should have a diacritic--i.e. a *Spanish-diacriticizer*.  

## Usage
  
**1.  To diacriticize a sequence of unidecoded Spanish tokens:**

- [ ] Open a new terminal and run the following:
     
      ~ % cd ~/Desktop 
          git clone https://github.com/elizabethgarza/nltk-Spanish-diacriticizer.git
          cd ~/Desktop/nltk-Spanish-diacriticizer/src
          ./diacriticize.py "Esta nina no esta viendo la pelicula espanola."
     
   You will get an output that looks something like this: 
      
          Esta niña no está viendo la película española.
        
 - [ ] Now try your own! 
 
       ~ %  ./diacriticize.py "Your own unidecoded sequence of tokens."   
     
## Background 

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

## Description of algorithm 

## Data collection

## Methodology

## Evaluation and error analysis

## #TODOs

**High priority**
  1. Preprocessing revisions: 
     - Make sure that both mellizas have the same number of sentences for all sets. 
     - Speed up preprocessing by refining for-loops.
  2. Split data into 70-10-10-10 train, micro_dev, micro_test, test_baseline sets.  
  3. Consider calculating micro-accuracies with k-folding. 
  4. Add Spanish punctuation to make output of `diacriticize.py` more consumer friendly.
  4. Consider making invariant dictionary more robust by systematically scraping *wiktionary*. 
  5. Add a set of grammar rules to account for invariantly diacriticized suffixes.
  6. Concatenate `Fisher` corpus of Spanish telephone conversations to data_set to add less academic sentences, or more data where needed.
  
**Future iterations of this project.** 
  1. Optimize training and prediction speeds by experimenting with Naive Bayes sk-learn classifiers. 
  2. Include POS tags and suffix endings as features in `extract features`.
  3. Perform the same experiment with neural networks.
  
  
  


