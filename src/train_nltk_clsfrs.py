import argparse
import nltk
import os
import pandas
import pickle
from tqdm import tqdm
import unidecode

import preprocess
import clsfr_prep


def mk_clsfr(corpus_path, melliza1, melliza2): 

    train_prep, test_prep = preprocess.Preprocess(corpus_path, melliza1, melliza2).mk_train_and_test_set()
    train = clsfr_prep.Clsfr_Prep(train_prep, test_prep).mk_nltk_train_set()
    test = clsfr_prep.Clsfr_Prep(train_prep, test_prep).mk_nltk_test_set()
    nltk_clsfr = nltk.classify.naivebayes.NaiveBayesClassifier.train(train)
    print(f'{unidecode.unidecode(melliza1)} accuracy:  {nltk.classify.util.accuracy(nltk_clsfr, test):.4f}')
    
    return nltk_clsfr

if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Trains nltk naive bayes classifiers and loads them into pickle files')
    parser.add_argument('corpus_path', help='str: path to corpus file')
    parser.add_argument('-no', help='int: number of classifiers that are to be trained')
    parser.add_argument('-m1_m2', help='str: melliza pair to be classified with dash in between; e.g. `él_el` ')
    args = parser.parse_args()

    # option to train on a specific number of clsfrs, rather than the default no., 200.
    if args.no: 
        os.chdir('..')
        os.chdir('data')
        # if you are making classifiers in batches, add argument `skiprows = [i for i in range(1,no_of_rows_to_skip)]` 
        mellizas = pandas.read_csv('top_200_mellizas.csv', sep=',', nrows=int(args.no))
        max1_max2_tokens = mellizas['MAX1_TOKEN'] + '\t'+ mellizas['MAX2_TOKEN'] 
        for row in tqdm(max1_max2_tokens): 
            tokenized_row = row.split()
            os.chdir('pickles')
            melliza1 = tokenized_row[0]
            melliza2 = tokenized_row[-1]
            saved_clsfr = open(f'{unidecode.unidecode(melliza1)}.pickle', 'wb')
            pickle.dump(mk_clsfr(args.corpus_path, melliza1, melliza2), saved_clsfr)
            saved_clsfr.close()
    # option to train a single clsfr
    elif args.m1_m2:
        os.chdir('..')
        os.chdir('data/pickles')
        melliza1 = args.m1_m2.split('_')[0]
        melliza2 = args.m1_m2.split('_')[-1]
        clsfr = mk_clsfr(args.corpus_path, melliza1, melliza2)
        saved_clsfr = open(clsfr, saved_clsfr)
        saved_clsfr.close()
    # default algorithm to classify top 200 mellizas.
    else:
        os.chdir('..')
        os.chdir('data')
        # if you are making classifiers in batches, add argument `skiprows = [i for i in range(1,23)]` if you want to, 
        # e.g. skip the first 23 rows.
        mellizas = pandas.read_csv('top_200_mellizas.csv', sep=',')
        max1_max2_tokens = mellizas['MAX1_MELLIZA'] + '\t'+ mellizas['MAX2_MELLIZA'] 
        for row in tqdm(max1_max2_tokens): 
            tokenized_row = row.split()
            os.chdir('pickles')
            melliza1 = tokenized_row[0]
            melliza2 = tokenized_row[-1]
            saved_clsfr = open(f'{unidecode.unidecode(melliza1)}.pickle', 'wb') 
            pickle.dump(mk_clsfr(args.corpus_path, melliza1, melliza2), saved_clsfr)
            saved_clsfr.close()






#parser = argparse.ArgumentParser()
#group = parser.add_mutually_exclusive_group()
#group.add_argument('-v', '--verbose', action='store_true')
#group.add_argument('-q', '--quiet', action='store_true')
#parser.add_argument('x', type=int, help='the base')
#parser.add_argument('y', type=int, help='the exponent')
#args = parser.parse_args()
#answer = args.x**args.y

#if args.quiet:
 #   print answer
#elif args.verbose:
 #   print '{} to the power {} equals {}'.format(args.x, args.y, answer)
#else:
#    print '{}^{} == {}'.format(args.x, args.y, answer)

        

