#!/usr/bin/env python3
"""Computes the proportions of mellizas and invariantly diacriticized tokens in a corpus."""


import argparse
import itertools
import os
from tqdm import tqdm 
import unidecode 

import diacriticize

if __name__=='__main__': 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus_path', help = 'str:  path to corpus; e.g.  `dev.txt` ')
    args = parser.parse_args()

              
    # instantiates a `Diacriticize` object that will attempt to diacriticize all undiacriticized tokens 
    diacriticizer = diacriticize.Diacriticize()

    os.chdir('..')
    os.chdir('data')

with open(args.corpus_path, 'r') as source: 

    # creates two lists: 
        # 1. a list of tokens from the original corpus, titled `original_toks` 
        # 2. a list of tokens from the diacriticized corpus, titled `predicted_toks`
    original_toks = []
    predicted_toks = []
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
            elif 'ificamente' in unidec_melliza[-10:]: 
                predicted_toks.append(unidec_melliza.replace('ificamente', 'íficamente'))
            elif 'rian' in unidec_melliza[-4:]: 
                predicted_toks.append(unidec_melliza.replace('rian', 'rían'))
            elif 'graficas' in unidec_melliza[-8:]: 
                predicted_toks.append(unidec_melliza.replace('graficas', 'gráficas'))    
            elif 'graficos' in unidec_melliza[-8:]: 
                predicted_toks.append(unidec_melliza.replace('graficos', 'gráficos'))
            elif 'grafico' in unidec_melliza[-7:]: 
                predicted_toks.append(unidec_melliza.replace('grafico', 'gráfico'))
            elif 'grafica' in unidec_melliza[-7:]: 
                predicted_toks.append(unidec_melliza.replace('grafica', 'gráfica'))                    
            else:
                predicted_toks.append(unidec_melliza)

    # computes percentage of mellizas that are correctly predicted
    
    ## puts all the mellizas that were used to train clsfrs into a list, titled 'pickles'
    os.chdir('pickles')
    pickles = os.listdir()
    picks = [pickle.replace('.pickle', '') for pickle in pickles]

    ## computes the total number of mellizas that were correctly and incorrectly diacriticized
    correct = 0 
    incorrect = 0
    for original_tok, predicted_tok in tqdm(zip(original_toks, predicted_toks)):
        if unidecode.unidecode(original_tok) in picks: 
            if original_tok == predicted_tok: 
                correct += 1
            else: 
                incorrect +=1 

    ## computes the total number of mellizas and stores in variable titled, `total_mells`, and prints the precent of mellizas that are correctly predicted 
    total_mells = correct + incorrect
    melliza_tok_acc = correct / total_mells
    print('\n' + '\n' + f'MELLIZA TOKEN ACCURACY')
    print('==============================================================')
    print(f'Correct:  {correct}')
    print(f'Incorrect:  {incorrect}')
    print(f'Total:  {total_mells}')
    print('==============================================================')
    print(f'Accuracy:  {round((correct / total_mells), 4)}'+ '\n'+ '\n')

    # computes percentage of invariantly diacriticized tokens that are correctly predicted 

    ## unidecodes (i.e. strips tokens of diacritics) tokens from original toks and appends those tokens to a list
    unidec_toks = []
    for tok in original_toks: 
        unidec_tok = unidecode.unidecode(tok)
        unidec_toks.append(unidec_tok)

    ## computes and prints the following: 
        # 1.  total number of tokens in original corpus that are diacriticized
        # 2.  total number of tokens in original corpus that are invariantly diacriticized
        # 3.  percentage of invariantly diacriticized tokens in the predicted corpus that are correctly diacriticized 
    diacriticized_toks = 0
    invariantly_diacriticized_toks = 0
    diacriticized_mellizas = 0
    correct = 0 
    incorrect = 0 
    for tok, unidec_tok, predicted_tok in zip(original_toks, unidec_toks, predicted_toks): 
        if tok != unidec_tok: 
            diacriticized_toks += 1
            if unidecode.unidecode(tok) in picks: 
                diacriticized_mellizas += 1
            if unidecode.unidecode(tok) not in picks:
                invariantly_diacriticized_toks += 1 
                if tok.casefold() == predicted_tok: 
                    correct += 1
                else: 
                    incorrect += 1
    invar_dia_tok_acc = (correct / (incorrect + correct)) 
    print(f'INVARIANTLY DIACRITICIZED TOKEN ACCURACY')
    print('==============================================================') 
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'Total:  {invariantly_diacriticized_toks}')
    print('==============================================================')
    print(f'Accuracy:  {round(invar_dia_tok_acc, 4)}' + '\n' + '\n')

    assert(diacriticized_mellizas + invariantly_diacriticized_toks == diacriticized_toks)

    # computes percentage of all diacriticized tokens that are correctly predicted 

    ## creates a set of all tokens that are diacriticized in either `original_toks` or `predicted_toks`
    diacriticized_tok_set = set()
    for tok, unidec_tok, predicted_tok in zip(original_toks, unidec_toks, predicted_toks): 
        if tok != unidec_tok: 
            diacriticized_tok_set.add(unidec_tok)
        if predicted_tok != unidec_tok: 
            diacriticized_tok_set.add(unidec_tok)

    ## computes percentage described above
    correct = 0 
    incorrect = 0
    for tok, unidec_tok, predicted_tok in zip(original_toks, unidec_toks, predicted_toks): 
        if unidec_tok in diacriticized_tok_set: 
            if tok == predicted_tok: 
                correct += 1 
            else: 
                incorrect += 1
    diacriticized_tok_acc = (correct / (incorrect + correct)) 
    print(f'DIACRITICIZED TOKEN ACCURACY:')
    print('==============================================================') 
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'Total: {correct + incorrect}')
    print('==============================================================') 
    print(f'Accuracy: {round(diacriticized_tok_acc, 4)}' + '\n' + '\n')

    # computes the baseline token accuracy, which is equal to the token accuracy of the text if the corpus were left undiacriticized .

    baseline_accuracy = (len(original_toks) - diacriticized_toks)/len(original_toks)
    print(f'BASELINE TOKEN ACCURACY')
    print('==============================================================') 
    print(f'Correct: {len(original_toks) - diacriticized_toks}')
    print(f'Incorrect: {diacriticized_toks}')
    print(f'Total: {len(original_toks)}')
    print('==============================================================') 
    print(f'Accuracy: {round(baseline_accuracy, 4)}' + '\n' + '\n')

    # computes the token accuracy, which is equal to the percentage of tokens in predicted corpus that match those in the original corpus

    correct = 0 
    incorrect = 0
    incorrect_toks = []
    for tok, predicted_tok in zip(original_toks, predicted_toks):
        if tok == predicted_tok: 
            correct += 1
        else: 
            incorrect += 1
            incorrect_toks.append(tok)
    per_correct_pred_toks = (correct / (incorrect + correct)) 
    print(f'TOKEN ACCURACY')
    print('==============================================================') 
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'Total: {len(original_toks)}')
    print('==============================================================') 
    print(f'Accuracy: {round(per_correct_pred_toks, 4)}' + '\n' + '\n')
 


















        
