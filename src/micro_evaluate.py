#!/usr/bin/env python3
"""Micro-evaluates a melliza dev set and prints errors to a file titled f`{unidecode.unidecode(melliza)}-errors.txt` for the purposes of error analysis.""" 


import argparse
import itertools
import os
from tqdm import tqdm 
import unidecode 

import diacriticize


if __name__=='__main__': 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('melliza', help = 'str:  unidecoded melliza that you wish to micro-evaluate for error analysis; e.g. como')
    args = parser.parse_args()

          
    # instantiates a `Diacriticize` object that will diacriticize all tokens that are identical to melliza
    diacriticizer = diacriticize.Diacriticize()

    os.chdir('..')
    os.chdir('data')
    os.chdir('micro_devs') 

    with open(f'{args.melliza}_dev.txt', 'r') as source: 
        
        # diacriticizes all mellizas per itemized sentences and appends predictions, along with all other tokens to a list, titled `predicted_toks`
        # also creates a separate list of tokens from the original development set, titled `original_toks`
        predicted_toks = []
        original_toks = []
        for sent in tqdm(source):
            tok_sent = sent.split()
            original_toks.append(tok_sent)
            for index, tok in enumerate(tok_sent):
                if unidecode.unidecode(tok) == args.melliza:
                    unidec_melliza = unidecode.unidecode(tok.casefold())
                    if unidec_melliza in diacriticizer.melliza_and_clsfr_dict:
                        predicted_toks.append(diacriticizer.predict_token(tok_sent, index, unidec_melliza))
                else:
                    predicted_toks.append(tok)

        # joins all itemized sentences to simplify micro-accuracy computation
        original_toks = list(itertools.chain.from_iterable(original_toks))

        # calculates micro-accuracies 
        correct = 0 
        incorrect = 0
        errors = []
        for index, (original_tok, predicted_tok) in enumerate(zip(original_toks, predicted_toks)): 
            if unidecode.unidecode(original_tok) == args.melliza: 
                if original_tok == predicted_tok: 
                    correct += 1
                else: 
                    incorrect += 1
                    try:
                        errors.append(f'ORIGINAL: {original_toks[index-5:index+5]}')
                        errors.append(f'PREDICTION: {predicted_toks[index-5:index+5]}\n')
                    except: 
                        pass 
        print(f'{args.melliza.upper()} micro-accuracy: {correct / (correct + incorrect) :4f}')

        # calculate a baseline accuracy
        correct = 0 
        incorrect = 0
        for original_tok, predicted_tok in zip(original_toks, predicted_toks): 
            if original_tok == predicted_tok: 
                correct += 1
            else: 
                incorrect += 1    
        print(f'{args.melliza.upper()} baseline accuracy: {correct / ( correct + incorrect ) :4f}')
        
        # prints phrases with errors to a file titled, {args.melliza}_errors.txt
        with open(f'{args.melliza}_errors.txt', 'w') as sink: 
            for error in errors: 
                print(f'{error}', file=sink)
            print(file=sink)
 
    

   







