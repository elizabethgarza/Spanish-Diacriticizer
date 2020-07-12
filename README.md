# nltk-Spanish-diacriticizer | Supervised machine learning using the Naive Bayes classification algorithm.

## Purpose 

The purpose of this project is to design model based on a supervised machine learning Naive Bayes classification algorithm that can predict whether or not a letter in a Spanish word should have a diacritic--i.e. a *Spanish-diacriticizer*.  

## Usage

Once you clone this repo, you will be able to do two things:

  1.  diacriticize a sequence of unidecoded Spanish tokens based on the trained model that is already available in this repo.
  2.  use your own Spanish data to train and evaluate your own model.
  
See **1.** and **2.** directly below for specific instructions on how to do these things.
  
**1.  To diacriticize a sequence of unidecoded Spanish tokens:**

- [ ] Open a new terminal and run the following:
     
      ~ % cd ~/Desktop 
          git clone https://github.com/elizabethgarza/nltk-Spanish-diacriticizer.git
          cd ~/Desktop/nltk-Spanish-diacriticizer/src
          python3 diacriticizer.py "Ella esta enojada."
     
   You will get an output that looks something like this: 
      
          Ella está enojada.
        
 - [ ] Now try your own! 
 
       ~ %  python3 diacriticizer.py "your own unidecoded sequence of tokens"   
     
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

## Evaluation and future work



