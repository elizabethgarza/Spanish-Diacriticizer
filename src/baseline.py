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
    parser.add_argument('corpus_path', help = 'str:  path to corpus that will be used for baseline evaluation')
    args = parser.parse_args()

          
    # instantiates a `Diacriticize` object that will diacriticize all tokens that are identical to melliza
    diacriticizer = diacriticize.Diacriticize()

    os.chdir('..')
    os.chdir('data')

    with open(args.corpus_path, 'r') as source: 
            
        # diacriticizes all mellizas per itemized sentences and appends predictions, along with all other tokens to a list, titled `predicted_toks`
        # also creates a separate list of tokens from the original development set, titled `original_toks`
        predicted_toks = []
        original_toks = []
        for sent in tqdm(source):
            tok_sent = sent.split()
            for tok in tok_sent: 
                tok = tok.casefold()
                original_toks.append(tok)
            for index, tok in enumerate(tok_sent):
                unidec_melliza = unidecode.unidecode(tok.casefold())
                if unidec_melliza in diacriticizer.melliza_and_clsfr_dict:
                    predicted_toks.append(diacriticizer.predict_token(tok_sent, index, unidec_melliza))
                elif unidec_melliza in diacriticizer.invariantly_diacriticized_tokens_dict: 
                    predicted_toks.append(diacriticizer.invariantly_diacriticized_tokens_dict[unidec_melliza])
                elif 'ion' in unidec_melliza[-3:]:
                    predicted_toks.append(unidec_melliza.replace('ion', 'ión'))
                elif 'patia' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('patia', 'patía'))
                elif 'patias' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('patias', 'patías'))  
                elif 'grafía' in unidec_melliza[-6:]: 
                    predicted_toks.append(unidec_melliza.replace('grafia', 'grafía'))
                elif 'grafías' in unidec_melliza[-7:]: 
                    predicted_toks.append(unidec_melliza.replace('grafias', 'grafías'))
                elif 'latria' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('latria', 'latría'))
                elif 'eria' in unidec_melliza[-4:]:
                    predicted_toks.append(unidec_melliza.replace('eria', 'ería'))
                elif 'erias' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('erias', 'erías'))
                elif 'latrias' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('latrias', 'latrías'))
                elif 'tomia' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('tomia', 'tomía'))
                elif 'tomias' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('tomias', 'tomías'))
                elif 'logia' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('logia', 'logía'))
                elif 'logias' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('logias', 'logías'))
                elif 'abamos' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('abamos', 'ábamos'))
                elif 'america' in unidec_melliza[-7:]:
                    predicted_toks.append(unidec_melliza.replace('america', 'américa'))
                elif 'americas' in unidec_melliza[-8:]:
                    predicted_toks.append(unidec_melliza.replace('americas', 'américas'))    
                elif 'logico' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('logico', 'lógico'))
                elif 'fobico' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('fobico', 'fóbico'))
                elif 'fobicos' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('fobicos', 'fóbicos'))
                elif 'logicos' in unidec_melliza[-8:]:
                    predicted_toks.append(unidec_melliza.replace('logicos', 'lógicos'))
                elif 'zon' in unidec_melliza[-3:]:
                    predicted_toks.append(unidec_melliza.replace('zon', 'zón'))
                elif 'zones' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('zones', 'zónes'))
                elif 'scopico' in unidec_melliza[-7:]:
                    predicted_toks.append(unidec_melliza.replace('scopico', 'scópico'))
                elif 'scopicos' in unidec_melliza[-7:]:
                    predicted_toks.append(unidec_melliza.replace('scopicos', 'scópicos'))
                elif 'onimo' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('onimo', 'ónimo'))
                elif 'onimos' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('onimos', 'ónimos'))
                elif 'onicos' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('onicos', 'ónicos'))
                elif 'onicas' in unidec_melliza[-6:]:
                    predicted_toks.append(unidec_melliza.replace('onicas', 'ónicas'))
                elif 'onica' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('onica', 'ónica'))
                elif 'onico' in unidec_melliza[-5:]:
                    predicted_toks.append(unidec_melliza.replace('onico', 'ónico'))
                elif 'ectomias' in unidec_melliza[-8:]: 
                    predicted_toks.append(unidec_melliza.replace('ectomias', 'ectomías'))
                elif 'ectomia' in unidec_melliza[-7:]: 
                    predicted_toks.append(unidec_melliza.replace('ectomia', 'ectomía'))
                elif 'cigotico' in unidec_melliza[-8:]: 
                    predicted_toks.append(unidec_melliza.replace('cigotico', 'cigótigo'))
                elif 'cigoticos' in unidec_melliza[-9:]: 
                    predicted_toks.append(unidec_melliza.replace('cigoticos', 'cigótigos'))     
                elif 'centrico' in unidec_melliza[-9:]: 
                    predicted_toks.append(unidec_melliza.replace('centrico', 'céntrico'))
                elif 'centricos' in unidec_melliza[-9:]: 
                    predicted_toks.append(unidec_melliza.replace('centricos', 'céntricos'))  
                elif 'aceo' in unidec_melliza[-4:]: 
                    predicted_toks.append(unidec_melliza.replace('aceo', 'áceo'))
                elif 'aceos' in unidec_melliza[-4:]: 
                    predicted_toks.append(unidec_melliza.replace('aceos', 'áceos'))
                elif 'orico' in unidec_melliza[-5:]: 
                    predicted_toks.append(unidec_melliza.replace('orico', 'órico'))
                elif 'oricos' in unidec_melliza[-5:]: 
                    predicted_toks.append(unidec_melliza.replace('oricos', 'óricos'))
                elif 'iendose' in unidec_melliza[-7:]: 
                    predicted_toks.append(unidec_melliza.replace('iendose', 'iéndose'))
                elif 'ificamente' in unidec_melliza[-7:]: 
                    predicted_toks.append(unidec_melliza.replace('ificamente', 'íficamente'))           
                else:
                    predicted_toks.append(unidec_melliza)

        # joins all itemized sentences to simplify micro-accuracy computation
        #original_toks = list(itertools.chain.from_iterable(original_toks))


        # calculate a baseline accuracy
        correct = 0 
        incorrect = 0
        errors = []
        for index, (original_tok, predicted_tok) in enumerate(zip(original_toks, predicted_toks)): 
            if original_tok.casefold() == predicted_tok: 
                correct += 1
            else: 
                incorrect += 1    
                try:
                    errors.append(f'ORIGINAL: {original_toks[index-5:index+5]}')
                    errors.append(f'PREDICTION: {predicted_toks[index-5:index+5]}\n')
                except: 
                    pass 
        print(f'BASELINE ACCURACY: {correct / ( correct + incorrect ) :4f}')
        
        # prints phrases with errors to a file titled, {args.melliza}_errors.txt
        with open(f'baseline_errors.txt', 'w') as sink: 
            for error in errors: 
                print(f'{error}', file=sink)
            print(file=sink)