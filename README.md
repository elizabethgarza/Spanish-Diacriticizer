# nltk-Spanish-diacriticizer | Supervised machine learning using the Naive Bayes classification algorithm.

## Purpose 

The purpose of this project is to design a model that can predict whether or not a character in a Spanish token should have a diacritic--i.e. to design a *Spanish-diacriticizer*.  

## Usage
  
**To diacriticize a sequence of unidecoded Spanish tokens:**

- [ ] Open a new terminal and run the following:
     
      ~ % cd ~/Desktop 
          git clone https://github.com/elizabethgarza/nltk-Spanish-diacriticizer.git
          cd ~/Desktop/nltk-Spanish-diacriticizer/src
          ./diacriticize.py "Esta nina no esta viendo la pelicula espanola."
     
   You will get an output that looks like this: 
      
          Esta niña no está viendo la película española.
        
 - [ ] Now try your own! 
 
       ~ %  ./diacriticize.py "Your own undiacriticized sequence of tokens."   
     
## Background 

For some undiacriticized words, like *espanol*, a diacriticizer model would have to perform a simple dictionary lookup to determine whether or not the *n* should have a diacritic. But for others, like *esta*, this operation will not suffice because there are two viable word forms stored in the value entry: *está* and *esta*. In such cases, such a model has to be given more information about the words before it can predict whether or not to add a diacritic to the letter *a*.  

Unlike computers, a Spanish speaking writer can determine which of the two words is correct by analyzing the word’s grammatical context.  For example, in this sentence--

   *Esa casa [esta] sucia. / That house [is] dirty.     (ex. 1)*

--a writer can quickly determine that the correct form of the word *esta* should be *está* because it functions as the verb, *is*, instead of the adjective *this*. 


Outside of a grammatical context, there are also three simple pronunciation-based rules which humans can rely on, as long as the word is not a homophone: 

1. Words ending in a vowel, -n, or -s are stressed on the next to the last (penultimate) syllable--e.g.: 
   *na-da* / *nothing* is pronounced **na**-*da*.

2. Words ending in any consonant except -n or -s are stressed on the last syllable.--e.g. 
   *doc-tor* / *doctor* is pronounced *doc*–**tor**

3. When rules #1 and #2 above are not followed, a written accent is used--e.g.
   *com-pró* / *bought* is pronounced *com*–**pró**; 
   *álbum* is pronounced **ál**–*bum*.

For a non-homophonic undiacritcized word like *calculo*, which can appear in three different forms {*calculo/calculation*, *cálculo/calculus*, *calculó/calculated*}, a Spanish speaker could also think about how the word is pronounced to determine the correct spelling.

Because the computer will only be able to interpret written text, the pronunciation-based rules described above will not be helpful for this task.  (Even if it could, using these rules in isolation would not be able to disambiguate frequently used homophones like {*si*, *sí*} / {*if* , *yes*}, {*cómo*, *como*} / {*how*, *like*}, {*sólo*, *solo*} / {*only*, *alone*}, etc.)  And while one could write an algorithm based on grammatical rules, such a system would quickly become too complex to be feasibly executed. 

One popular alternative which I've decided to use for this project is the Naive Bayes classification algorithm, which can learn to make predictions without explicit instruction. Aside from designing a model that correctly adds diacritics to undiacriticized words in a sentence, the other purpose of this project is to explore the pros and cons of using both the NLTK and Scikit multinomial versions of this algorithm, the latter of which I will post at a later date.

## Description of algorithm 

### General description of how naive Bayes applies to this task

Classification problems in machine learning consist of classifying, or assigning labels to "ambiguous linguistic signals or events" (from Lecture 8 on http://wellformedness.com/courses/LING83800). In *ex. 1* above, the model would have to decide whether to assign either label_1, *esta*, or label_2, *está*, to what would be considered an "ambiguous linguistic event". The model makes this decision by recalling what sentence features tend to be associated with either label by calculating probabilities.  

To calculate the probability of classifying label_1, or *L1* given one specific feature, the computer calculates--

![Image 7-22-20 at 12 40 PM](https://user-images.githubusercontent.com/43279348/88203984-a2277e00-cc18-11ea-9a21-4214ed176e7d.jpg)

--where, *L1* refers to *esta*, and *sf* refers to any given sentence feature--e.g., sentence length.  (To calculate *P(L2 | sf)* one would simply have to subtract *P(L1 | sf)* from *1*.)  Calculating the probability of *L1*, given all sentence features, or *{SF}*, simply involves multiplying all probabilities of *L1*, given a particular sentence feature, like so-- 

![Image 7-22-20 at 12 43 PM](https://user-images.githubusercontent.com/43279348/88204376-2aa61e80-cc19-11ea-8496-ea3f7a044d11.jpg)

The model is considered naive because, “it assumes that the probability of each feature...is conditionally independent of every other feature” (Lecture 8), meaning that the value for *P(sfi | L1)* in *eq. 2* above would actually be--

![Image 7-22-20 at 12 47 PM](https://user-images.githubusercontent.com/43279348/88204629-8bcdf200-cc19-11ea-95b8-ca478da7dc6c.jpg)

--were it not for this assumption.  Indeed, although eq. 2 does simplify the calculations needed to make a prediction, it ignores the conditional interdependence of different sentence features, which likely decreases the overall accuracy of the predictions made by Naive Bayes.

### Sentence features, or *{SF}*

Examples of some relevant sentence features could include:  the type of punctuation used, the length of the sentence, what parts of speech tend to precede and follow the word *esta*, etc. Ultimately, it is up to the analyst to decide which features to focus on.  For this project, I decided to start with word context features--i.e. which words occur up to four places to the left and right of the label.  Other  miscellaneous features include whether or not the label appears at the beginning or end of the sentence, and whether or not the label appears in isolation.  For instance, for the sentence and label pair-- 

  *('El perro no está aquí.', 'está') / ('The dog is not here.', 'is')  (ex. 2)*

--the sentence features, *{SF}*, would be:  

![Image 7-23-20 at 12 26 PM](https://user-images.githubusercontent.com/43279348/88312341-05c8ae80-cce0-11ea-94cf-1cc319fd1474.jpg)

## Description of data

A 2.2 GB wikipedia corpus of ~19 million lines was used for this experiment.  For instructions on how to download the corpus, please refer to https://github.com/elizabethgarza/nltk-Spanish-diacriticizer/blob/master/data/instructions_to_download_full_data_set.

### Description and distribution of token types in the data set.

Bolded below you will see a description of each of the token types found in the data:

- **Mellizas** refer to fraternal token twins like  {*el*, *él*} / {*the*, *he*} .  While most of these words are homographs (words that have different pronunciations for the same spelling like *live*), this term is useful for this project because a few mellizas are homophones (words that have the same pronunciation, but different meanings) like {*si*, *sí*} / {*if*, *yes*}, {*cómo*, *como*} / {*how* , *like*} and {*sólo*, *solo*} / {*only* , *alone*}.  In sum, the elements within melliza sets can be formally defined as--

  *{x | x is a word that if undiacriticized would look identical to all other undiacriticized x’s}.*  

- **Undiacriticized tokens** are tokens that don't have diacritics, like {*libro*, *perro*, *gato*, *el*, *esta*...} / {*book*, *dog*, *cat*, *the*, *this*}, which includes undiacriticized mellizas.

- **Diacriticized tokens** are tokens that have a diacritic, like {*lápiz*, *película, *español*, *él*, *está*} / {*pencil*, *movie*, *Spanish*, *he*, *is*}, which includes diacriticized mellizas.  

- **Invariantly diacriticized tokens** are tokens that always have a diacritic, like {*lápiz*, *película*, *español*} / {*pencil*, *movie*, *Spanish*}, which excludes all mellizas. 

The figure below shows the distribution of these token types.  Of particular note is that diacriticized tokens constitute ~10% of all tokens in a given set.  And of that 10%, only ~12% of those tokens are mellizas; the other ~88% are invariantly diacriticized tokens.

**Figure 1. An illustration of the distribution of token types in the data**
![Image 7-22-20 at 4 41 PM](https://user-images.githubusercontent.com/43279348/88226540-3524e000-cc3a-11ea-89d9-bc0354956a69.jpg)

### Melliza statistics.

- A total of 3710 mellizas were extracted from the corpus after eliminating typos and all non-Spanish mellizas. 
- The table in **Figure 2** below shows the frequency counts for an excerpt of the *pandas* dataframe that contains that top 200 most frequently occuring mellizas in the data.  

**Figure 2** Top 200 most frequently occuring mellizas in the data.  
![Image 7-23-20 at 7 35 PM](https://user-images.githubusercontent.com/43279348/88348624-ed768500-cd1b-11ea-8618-d9e35e737f90.jpg)

- The table in **Figure 3** below provides summary statistics, which reveal that mean total frequency is ~95K and that the mode is 9601, which is the total frequency count that you see at the very bottom of the table in **Figure 2** above.  In all, these summary statistics reveal that training around 200 classifiers will cover the bulk--but not all--of the mellizas that will need to be diacriticized in the data.

**Figure 3** 

![Image 7-23-20 at 7 35 PM-1](https://user-images.githubusercontent.com/43279348/88348911-ab017800-cd1c-11ea-8e63-d1de685d6c6f.jpg)
 
## Methodology

*Pre-processing phase* 

- [ ] Shuffle the lines in the corpus and split into `train`, `dev`, and `test` sets.  The split should adhere to a 90-5-5 proportionality.
- [ ] Extract mellizas from the corpus, and compute the frequencies of both diacriticized and undiacriticized mellizas in the data.  Make a table of the top 200 mellizas with the highest average frequencies. 
- [ ] For each of the 200 mellizas, shuffle the lines in `train` from Step 1 and split into `micro_train`, `micro_dev`, and `micro_test`.  Each split should adhere to an 80-10-10 proportionality. 
- [ ] Create a dictionary populated with invariantly diacriticized tokens by extracting tokens from any published Spanish texts that have been edited, along with any reputable lexicons.  The dictionary in this experiment, titled `invars.json` has thus far extracted tokens from the `Santiago Spanish lexicon` and the `fisher-CALLHOME` corpus, which contains transcripts of Spanish telephone conversations.  With those two sources, a dictionary with ~9000 entries has been created.
- [ ] Preprocess `micro_train` by doing the following to each of the sentences in the set: 
  - tokenizing each sentence
  - labelling each sentence as either an undiacriticized melliza, or a diacriticied melliza
  - extracting sentence features, *{SF}*, from each sentence

*Training phase*

- [ ] For each melliza, train a Naive Bayes classifier on the preprocessed data from Step 4 and store that classifier into a pickle file for later use.  Then, populate a melliza dictionary with keys and entries consisting of undiacriticized mellizas and their corresponding classifiers, respectively, using `train.py`.

*Development phase* 

- [ ] Strip diacritics from only melliza tokens in `micro_dev` and diacriticize them.  Then, compute baseline and token accuracies and print all errors to a separate file, using `train.py`.   
- [ ] Extract undiacriticized suffix forms from the errors found in separate files.  Then, populate a dictionary with keys and entries for undiacriticized suffixes and their diacriticized counterparts, respectively. (You can write a script to automate this process, but I think that a manual extraction is probably easier.)
- [ ] Strip diacritics from all tokens in `dev` and diacriticize them. Then, compute the melliza token accuracy, invariantly diacriticized token accuracy, diacriticized token accuracy, baseline token accuracy, and token accuracy, using `evaluate.py`.
- [ ] **Edit .py scripts, populate the `invars.json` dictionary with more entries, make new classifiers, and retrain current classifiers as needed to maximmize accuracies.**  (<------Currently at this stage!)

*Testing phase*

- [ ] Strip diacritics from only melliza tokens in `micro_test` and diacriticize them.  Then, compute baseline and token accracies, using `micro-evaluate.py`.
- [ ] Strip diacritics from all tokens in `test` and diacriticize them.  Then, compute the melliza token accuracy, invariantly diacriticized token accuracy, etc, using `evaluate.py`. 

## Evaluation and error analysis

As I have indicated above, I'm still in the development phase of this experiment, so final evaluation results will be posted at a later time.  For the time being, though, I have posted the evaluation results for `Dev` for your reference.

### Dev set accuracies
![Image 7-22-20 at 5 08 PM](https://user-images.githubusercontent.com/43279348/88229011-0dd01200-cc3e-11ea-8eeb-3067bdad8124.jpg)

### Development phase error analysis
The evaluation results above indicate that the invariantly diacriticized tokens are the most troublesome token type to correctly diacriticize, as the low accuracy of ~71% suggests.  The cause of this low accuracy is very obvious, but before I explain what it is, let me give you an outline of the decision making process that the model has to go through before it decides whether or not a token should be diacriticized.  

If the model is asked to diacriticize every token in the undiacriticized sentence, *El nino compro un lapiz.* / *The boy bought a pencil.*, it will iterate through each of the tokens in the sentence by asking the same series of questions for each token.  For the first word, *el*, for example, it will ask: 

        1. Is 'el' in the melliza dictionary? 
            If 'yes', use its classifier to decide whether or not 'el' should be diacriticized. Then, replace 'el' with its diacriticized form.
            If 'no', continue to the next question. 
        2. Is 'el' in the invariantly diacriticized token dictionary? 
            If 'yes', replace 'el' with diacriticized form.
        3. Does 'el' contain any invariantly diacriticized suffixes that can be found in the variantly diacriticized suffix dictionary?
            If 'yes' replace the current suffix with invariantly diacriticzed suffix.
            If 'no', leave 'el' as it is. 
            
After it runs through this series of questions for *el*, the model will ask them again for each of the other tokens in the sentence until it reaches the end.  To clarify, the "melliza dictionary" is a dictionary that contains keys and value entries that consist of the top 200 mellizas and their corresponding classifiers; the "invariantly diacriticized token dictionary" contains keys and value entries that consist of undiacriticized tokens and their diacriticized forms, respectively; and the "invariantly diacriticized suffix dictionary" contains keys and value entries that consist of undiacriticized suffixes and their diacriticized forms, respectively.

Given this decision making process, the reason the invariantly diacriticized token accuracy is low is because it is impossible to quickly populate the invariant dictionary with all the invariantly diacriticized tokens that currently exist.  For example, if you ask the model to diacriticize *Agarra ese panal.* / *Grab that diaper.*, when it gets to the third token, *panal*, it will advance to the second question above, find *panal* in the dictionary, and give you the correct output, namely *Agarra ese pañal*.  But if you ask it to do the same on *Agarra esos panales.* / *Grab those diapers.*, it will give you the wrong output: *Agarra esos panales.*  Reason being is that *pañal* happened to be found in the lexicon and corpus that I used to populate that dictionary, while its plural form, *pañales*, wasn't.  Altogether, the chances that I'll be able manually populate a dictionary with all forms of every invariantly diacriticized token that currently exists are null. This example in partidular is a testament to the immense amount of labor that would be involved in populating that dictionary until it robust enough to significantly improve overall accuracy.  

## Future work before evaluating the test set.

To obviate this dilemma, my current plan is to create two more classifiers.  The first, which will be called, `classify_unknowns` will be trained on a set of invariantly diacriticized and undiacriticized tokens, to the exclusion of mellizas.  Each token will receive a label that indicates which vowel is diacriticized, if any at all.  For example, a token like *lápiz* / *pencil* will receive the label *1* because the first vowel is diacriticized; *película* / *movie* will be labelled with a *2* because the second vowel is diacriticized; and *perro* / *dog* will be *0* because there are no diacriticized vowels, etc. This system of labelling will ask the model to classify words according to which syllable tends to be diacriticized, given a particular set of features, like the ones I described above in the section titled **Sentence features, or {SF}**.

The second classifier, called `classify_ns` will be trained on all tokens with either *n* or *ñ* so that the model will be able to make a good prediction when presented with tokens like *enseno*, which should be *enseñó* / *taught*.  

As such, the revised decision-making process for a token like *panales* will look something like this: 

     1. Is 'panales' in the melliza dictionary? 
            If 'yes', use its classifier to decide whether or not it should be diacriticized. Then, replace 'panales' with its diacriticized form. 
            If 'no', continue to the next question. 
     2. Is 'panales' in the invariantly diacriticized token dictionary? 
            If 'yes', replace 'panales' with its diacriticized form.
            If 'no', continue to the next question. 
     3. Does 'panales' contain any of the invariantly diacriticized suffixes that can be found in the invariantly diacriticized suffix dictionary?
            If 'yes', replace suffix with invariantly diacriticzed suffix.
            If 'no', leave 'panales' as it is. 
     4. Does 'panales' have an 'n'? 
            If 'yes', use the `classify_ns` to decide whether or not the `n` should be diacriticized.  Then, continue on to the next question. 
     5. Does 'panales' have any vowels? 
            If 'yes', use the `classify_unknowns` to decide which vowel, if any, should be diacriticized and replace 'panales' with the prediction. 
            If 'no', leave 'panales' as it is. 

Ultimately, my hope is that training two additional classifiers of this sort will lessen the diacriticizer's dependence on the dictionaries described above. 

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
  

  
  


