import argparse
import nltk
import os
import pandas
import pickle
from tqdm import tqdm
import unidecode

import preprocess
import clsfr_prep


def mk_clsfr(corpus_path, token1, token2): 

    train_prep, test_prep = preprocess.Preprocess(args.corpus_path, token1, token2).mk_train_and_test_set()
    train = clsfr_prep.Clsfr_Prep(train_prep, test_prep).mk_nltk_train_set()
    test = clsfr_prep.Clsfr_Prep(train_prep, test_prep).mk_nltk_test_set()
    nltk_clsfr = nltk.classify.naivebayes.NaiveBayesClassifier.train(train)
    print(f"{unidecode.unidecode(token1)} accuracy:  {nltk.classify.util.accuracy(nltk_clsfr, test):.4f}")
    
    return nltk_clsfr

if __name__ =="__main__": 
    parser = argparse.ArgumentParser(description="Trains nltk naive bayes classifiers and loads them into pickle files.")
    parser.add_argument("corpus_path", help="str: path to corpus file")
    parser.add_argument("no_of_clsfrs", help="int: number of classifiers that you want to train; recommended amount: 200")
    args = parser.parse_args()

    os.chdir('..')
    os.chdir('data')
    nrows = int(args.no_of_clsfrs)
    # changes the rows that you want to iterate over by adding the argument, e.g. skiprows=(1, 23, 1)
    mellizas = pandas.read_csv("top_200_mellizas.csv", sep=",", skiprows = [i for i in range(1, 23)], nrows=nrows)
    max1_max2_tokens = mellizas["MAX1_TOKEN"] + "\t"+ mellizas["MAX2_TOKEN"] 
    for row in tqdm(max1_max2_tokens): 
        tokenized_row = row.split()
        os.chdir('pickles')
        saved_clsfr = open(f"{unidecode.unidecode(tokenized_row[0])}.pickle", "wb") 
        pickle.dump(mk_clsfr(args.corpus_path, tokenized_row[0], tokenized_row[-1]), saved_clsfr)
        saved_clsfr.close()

        

