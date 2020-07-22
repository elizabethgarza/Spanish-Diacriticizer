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

   *Esa casa [esta] sucia. / That house [is] dirty.     (ex. 1)*

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

### General description of how naive Bayes applies to this task

Classification problems in machine learning consist of classifying, or assigning labels to "ambiguous linguistic signals or events" (from Lecture 8 on http://wellformedness.com/courses/LING83800). In the *ex. 1* in **Background**, the model would have to decide whether to assign either label_1, *esta*, or label_2, *está*, to what would be considered an "ambiguous linguistic event". The model makes this decision by recalling what sentence features tend to be associated with either label by calculating probabilities.  

To calculate the probability of classifying label_1 given one specific feature, the computer calculates--

![Image 7-22-20 at 12 40 PM](https://user-images.githubusercontent.com/43279348/88203984-a2277e00-cc18-11ea-9a21-4214ed176e7d.jpg)

--where, *L1* refers to *esta*, and *sf* refers to any given sentence feature, say, sentence length.  (To calculate *P(L2 | sf)* one would simply have to subtract *P(L1 | sf)* from *1*.)  Calculating the probability of *L1*, given all sentence features, or *{SF}*, simply involves multiplying all probabilities for *L1*, given a particular sentence feature, like so-- 

![Image 7-22-20 at 12 43 PM](https://user-images.githubusercontent.com/43279348/88204376-2aa61e80-cc19-11ea-8496-ea3f7a044d11.jpg)

The model is considered naive because, “it assumes that the probability of each feature...is conditionally independent of every other feature” (Lecture 8), meaning that the value for *P(sfi | L1)* in *eq. 2* above would actually be--

![Image 7-22-20 at 12 47 PM](https://user-images.githubusercontent.com/43279348/88204629-8bcdf200-cc19-11ea-95b8-ca478da7dc6c.jpg)

--were it not for this assumption.  Indeed, although eq. 2 does simplify the calculations needed to make a prediction, it ignores the conditional interdependence of different sentence features, which likely decreases the overall accuracy of the predictions made by Naive Bayes.

### Word context features

Examples of some relevant sentence features could include:  the type of punctuation used, the length of the sentence, what parts of speech tend to precede and follow the word *esta*, etc. Ultimately, it is up to the analyst to decide which features to focus on.  For this project, I decided to start with word context features, that is, which words occur up to four places to the left and right of the label.  Other  miscellaneous features include whether or not the label appears at the beginning or end of the sentence, and whether or not the label appears in isolation.  For instance, for the sentence and label pair-- 

![Image 7-22-20 at 12 50 PM](https://user-images.githubusercontent.com/43279348/88204904-fc750e80-cc19-11ea-82db-3511cdbda23a.jpg)

--the sentence features, {SF}, would be:  

![Image 7-22-20 at 12 51 PM](https://user-images.githubusercontent.com/43279348/88205043-33e3bb00-cc1a-11ea-8c63-e396941476a1.jpg)

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
![Image 7-22-20 at 4 41 PM](https://user-images.githubusercontent.com/43279348/88226540-3524e000-cc3a-11ea-89d9-bc0354956a69.jpg)

### Melliza statistics.

- A total of 3710 mellizas were extracted from the corpus.

#TODO: 
 - [ ] provide a histogram of frequency distribution
 - [ ] provide a snapshop of the csv table
 
## Methodology

*Pre-training phase* 

Step 1:  Shuffle the lines in the corpus and then split into train, dev, and test sets 
Step 2:  Extract all mellizas from the corpus.  Compute the frequencies of both diacriticized and undiacriticized mellizas in the data.  Print that data into a csv file for future reference.
Step 3:  Train classifiers on the top 200 mellizas with the average highest frequencies.
Step 4:  


## Evaluation and error analysis

### Dev set evaluation 
![Image 7-22-20 at 5 08 PM](https://user-images.githubusercontent.com/43279348/88229011-0dd01200-cc3e-11ea-8eeb-3067bdad8124.jpg)

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
  

  
  


