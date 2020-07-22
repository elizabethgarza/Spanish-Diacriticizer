# nltk-Spanish-diacriticizer | Supervised machine learning using the Naive Bayes classification algorithm.

## Purpose 

The purpose of this project is to design model based on a supervised machine learning Naive Bayes classification algorithm that can predict whether or not a letter in a Spanish word should have a diacritic--i.e. a *Spanish-diacriticizer*.  

## Usage
  
**To diacriticize a sequence of unidecoded Spanish tokens:**

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

For some unidecoded words, like "espanol", the model simply has to perform a dictionary lookup to determine whether or not the *n* should have a diacritic. But for others, like *esta*, this operation will not suffice because there are two viable word forms stored in the value entry: *está* and *esta*. In such cases, the model has to be given more information about the words before it can predict whether or not to add a diacritic to the letter *a*.  

Unlike computers, humans would quickly be able to decide which of the two words is correct by analyzing the word’s grammatical context.  For example, in this sentence--

    Esa casa [esta] sucia. / That house [is] dirty.

--a writer can quickly determine that the correct form of the word "esta" should be "está" because it functions as the verb, *is*, instead of the adjective *this*. 


Outside of a grammatical context, there are also three simple pronunciation-based rules which humans can rely on, as long as the word is not a homophone: 

1. Words ending in a vowel, -n, or -s are stressed on the next to the last (penultimate) syllable--e.g.: 
   *na-da* is pronounced **na**-*da*.

2. Words ending in any consonant except -n or -s are stressed on the last syllable.--e.g. 
   *doc-tor* is pronounced *doc*–**tor**

3. When rules #1 and #2 above are not followed, a written accent is used--e.g.
   *com-pró* is pronounced *com*–**pró**; 
   *álbum* is pronounced **ál**–*bum*.

For a non-homophonic unidecoded word like *calculo*, which can appear in three different forms {*calculo*, *cálculo*, *calculó*}, humans could also think about how the word is pronounced to determine the correct spelling.

Because the computer will only be able to interpret written text, the pronunciation-based rules described above will not be helpful for this task.  (Even if it could, using these rules in isolation would not be able to disambiguate frequently used homophones like {*si*, *sí*}, {*cómo*, *como*}, {*sólo*, *solo*}, etc.)  And while one could write an algorithm based on grammatical rules, such a system would quickly become too complex to be feasibly executed. 

One popular alternative which I've decided to use for this project is the Naive Bayes classification algorithm, which can learn to make predictions without explicit instruction. Aside from designing a model that correctly adds diacritics to unidecoded words in a sentence, the other purpose of this project is to explore the pros and cons of using both the NLTK and Scikit multinomial versions of this algorithm, the latter of which I will post at a later date.

## Description of algorithm 

#TODO fill this in.

## Description of data

A 2.2 GB wikipedia corpus of ~19 million lines was used for this experiment.  For instructions on how to download the corpus, please refer to https://github.com/elizabethgarza/nltk-Spanish-diacriticizer/blob/master/data/instructions_to_download_full_data_set.

### Description and distribution of token types in the data set.

Bolded below you will see a description of each of the token types found in the data:

- **Mellizas** refer to fraternal token twins like  {*el*, *él*}.  While most of these words are homographs (words that have different pronunciations for the same spelling like *live*), this term is useful for this project because a few mellizas are homophones (words that have the same pronunciation, but different meanings) like {*si*, *sí*}, {*cómo*, *como*} and {*sólo*, *solo*}.  In sum, the elements within melliza sets can be formally defined as--

{x | x is a word that if undiacriticized would look identical to all other x’s}  

--meaning that all mellizas would look identical if they appeared in their undiacriticized forms.

- **Undiacriticized tokens** refer to tokens that never have diacritics, like {*libro*, *perro*, *gato*, *el*, *esta*...}, which includes undiacriticized mellizas.

- **Diacriticized tokens** refer to any token that has a diacritic, like {*lapiz*, *pelicula*, *espanol*, *el*, *esta*}, which includes diacriticized mellizas.  

- **Invariantly diacriticized tokens** refer to any token that always has a diacritic, like {*lapiz*, *pelicula*, *espanol*}, which excludes all mellizas. 

The figure below uses a venn diagram to show the distribution of these token types.  Of particular note is that diacriticized tokens constitute ~10% of all tokens in a given set.  Of that 10%, only ~12% of those tokens are mellizas; the other ~88% are invariantly diacriticized tokens, as you can see below. 

**Figure 1. A venn diagram showing the distribution of token types in the data**
![Image 7-22-20 at 1 51 AM](https://user-images.githubusercontent.com/43279348/88139573-f8b39e80-cbbd-11ea-9a0a-8295e99d8589.jpg)

### Melliza statistics.

- A total of 3710 mellizas were extracted from the corpus.

#TODO: 
 - [ ] provide a histogram of frequency distribution
 - [ ] provide a snapshop of the csv table
 
## Methodology

#TODO fill this in. 

## Evaluation and error analysis

### Dev set evaluation 

![Image 7-21-20 at 10 15 PM](https://user-images.githubusercontent.com/43279348/88134660-2e06bf00-cbb3-11ea-912a-99e4b5c9cd67.jpg)

## Future work 

  - [ ] Optimize training and prediction speeds running the same experiment with multinomial Naive Bayes sk-learn classifiers . 
  - [ ] Train a Spanish POS tagger and include tags and suffix endings as features in `extract_features`.
  - [ ] Train binary classifiers on melliza suffixes?
  - [ ] Perform the same experiment with neural networks.
  - [ ] Package everything up if you can achieve a baseline accuracy above 98%.

## #TODOs

**High priority**

- On branch = 'csv-cleanup':
  - [ ] write an optional argument that prints evaluation results to either 'DEV_ACCURACY' or 'TEST_ACCURACY' on the csv file
  - [ ] add the two columns and number the entries

- On branch = 'rerun-csv-code': 
  - [ ] re-run code to regenerate CSV file without the bad entries 
  - [ ] add a snippet to print out a histogram of melliza distribution 
  - [ ] update the data section of READ.ME 
   
**Medium priority**
  - [ ] Add optional argparse argument to `diacriticize.py`:
     - Allow user to diacriticize a file via a `args.file_path` argument.
  - [ ] Add a set of grammar rules to account for invariantly diacriticized suffixes.
  - [ ] Consider calculating micro-accuracies with k-folding. 
  - [ ] Write a clean script for extracting mellizas from any given corpus so that users can train on their own corpus.
  - [ ] Make .py scripts more readable by import `typing`, and adding comments and descriptions to functions.
  
**Low priority**
  - [ ] Add Spanish punctuation to make output of `diacriticize.py` more consumer friendly.
  - [ ] Consider making invariant dictionary more robust by systematically scraping *wiktionary*. 
  - [ ] Optimize `extract_features` function.
  

  
  


