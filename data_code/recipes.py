# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.7.3 (default, Jan 22 2021, 20:04:44) 
# [GCC 8.3.0]
# Embedded file name: /raid/data/cennet/code/e2e/recipe.py
# Compiled at: 2022-05-17 23:07:08
# Size of source mod 2**32: 9057 bytes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os, io, re, attr, random
from fnmatch import fnmatch
from copy import deepcopy as c
from boltons.iterutils import pairwise
from .utils import *
from .ar_en_recipe_parser import get_en_recipe_data
from .ar_de_recipe_parser import  get_de_recipe_data
from .ar_tr_recipe_parser import get_tr_recipe_data


class Recipe:

    def __init__(self, video_id, parsed_recipe):
        self.video_id = video_id
        self.parsed_recipe = parsed_recipe
        self.steps, self.step_ids, self.tokens, self.indexes = [], [], [], []
        self.mentions = []
        self.mention_start_end = []
        self.nulls = []
        self.extract_recipe_features()

    def __getitem__(self, idx):
        return (
         self.tokens[idx], self.corefs[idx], self.speakers[idx], self.genre)

    def __repr__(self):
        return 'Recipe containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)


    def extract_recipe_features(self):
        for step_index, step in self.parsed_recipe.items():
            step_tokens = step['annot'].split()
            step_token_indexes = [idx for idx in range(len(self.tokens), len(self.tokens) + len(step_tokens))]
            
            self.steps.append(step_tokens)
            self.step_ids.append(step['step_id'])
            self.tokens += step_tokens
            self.indexes += step_token_indexes

            for obj in step['objects']:
                
                if obj['string'] not in ['null', 'NULL']:
                    obj_id = obj['id']
                    obj_string = obj['string']
                    
                    step_start_end_indexes = find_index_of_sublist(obj_string.split(), step_tokens)
                    
                    if step_start_end_indexes:
                        start = step_token_indexes[step_start_end_indexes[0]]
                        end = step_token_indexes[step_start_end_indexes[1]]
                        
                        self.mention_start_end.append((start, end))
                        self.mentions.append({'id':obj['id'],
                                              'step_index': step_index,
                                              'start':start,  
                                              'end':end,  
                                              'step_id': step['step_id'],
                                              'string':obj_string,
                                              'antecedents':obj['reference'], 
                                              'relation':obj['relation'],
                                              'is_step':0,
                                              'is_null':0})
                else:
                    start = step_token_indexes[0]
                    end = step_token_indexes[0]
                    self.mention_start_end.append((start, end))
                    self.mentions.append({'id':obj['id'],
                                          'step_index': step_index,
                                          'start':start,  
                                          'end':end,
                                          'step_id': step['step_id'],
                                          'string':'NULL',
                                          'antecedents':obj['reference'],
                                          'relation':obj['relation'],
                                          'is_step':0,
                                          'is_null':1})

            self.mention_start_end.append((step_token_indexes[0], step_token_indexes[-1]))
            self.mentions.append({'id':step['step_id'],
                                  'step_index': step_index,
                                  'start':step_token_indexes[0],  
                                  'end':step_token_indexes[-1],
                                  'step_id': step['step_id'], 
                                  'string':step['annot'],
                                  'antecedents':[], 
                                  'relation':'None',
                                  'is_step':1,
                                  'is_null':0})

class Corpus:
    def __init__(self, language):
        self.language = language
        self.parsed_recipes = {}
        self.recipes = self.get_corpus()

    def __getitem__(self, idx):
        return self.recipes[idx]

    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.recipes)

    def get_corpus(self):
        recipes = {}
        if self.language == 'en':
            recipes_data = get_en_recipe_data()
            self.parsed_recipes = recipes_data['recipes']
            
            for ind, (video_id, recipe) in enumerate(self.parsed_recipes.items()):
                recipe_object = Recipe(video_id, recipe)
                recipes[video_id] = recipe_object

        elif self.language == 'tr':
            recipes_data = get_tr_recipe_data()
            self.parsed_recipes = recipes_data['recipes']

            for ind, (video_id, recipe) in enumerate(self.parsed_recipes.items()):
                recipe_object = Recipe(video_id, recipe)
                recipes[video_id] = recipe_object

        
        elif self.language == 'de':
            recipes_data = get_de_recipe_data()
            self.parsed_recipes = recipes_data['recipes']

            for ind, (video_id, recipe) in enumerate(self.parsed_recipes.items()):
                recipe_object = Recipe(video_id, recipe)
                recipes[video_id] = recipe_object

        else:
            print('wrong language input: ', self.language)

        return recipes


def get_recipes(language):
    corpus = Corpus(language=language)
    print(len(corpus.recipes), language, ' recipes loaded...')
    return corpus

if __name__ == '__main__':
    #en_recipes = get_recipes('en')
    #tr_recipes = get_recipes('tr')
    de_recipes = get_recipes('de')