
import csv
import os
import math
import re
import random
import collections
import glob as glob
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import groupby
from operator import itemgetter

from pathlib import Path
import sys
#sys.path.insert(0, '/netscratch/oguz/impress/mmar/md_ar/github')
from data_code.recipes import *
from data_code.spans import *
from utils import *

from model import ResolutionModel

# language_vectors=language_vectors, visual_vectors=visual_vectors,
class Trainer:
    def __init__(self, en_corpus, tr_corpus, de_corpus, language_vectors, visual_vectors,
                       is_gold, learning_rate, anaphora_epochs, languages):

        self.__dict__.update(locals())
        self.en_corpus = en_corpus
        self.tr_corpus = tr_corpus
        self.de_corpus = de_corpus
        self.language_vectors = language_vectors
        self.visual_vectors = visual_vectors

        self.is_gold = is_gold
        self.anaphora_epochs = anaphora_epochs
        self.languages = languages

        self.video_ids = []
        with open('./datasets/train_video_ids.txt', 'r') as video_id_read_file:
            lines = video_id_read_file.readlines()
            for line in lines:
                self.video_ids.append(line.replace('\n',''))

        model = ResolutionModel()
        self.model = to_cuda(model)
        self.md_loss_function = nn.BCELoss()
        self.ar_loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        date = str(self.anaphora_epochs)+'_'+datetime.datetime.now().strftime('%d_%m_%y_%H_%M')
        trained_models_folder = './trained_folder' 
        if not os.path.isdir(trained_models_folder):
            os.mkdir(trained_models_folder)
        span =  'gold' if self.is_gold else 'candidate'
        self.model_name = span+'_'+date
        self.model_path = os.path.join(trained_models_folder, self.self.model_name)
        print()
        print('The ', self.model_name, 'model will be stored in:')
        print(trained_models_folder)
        print('learning_rate: ', self.learning_rate)
        print()


    def train(self):
        """ Train a model """
        ar_epoch_loss = 0.0
        ar_epochs = ['ar'] * self.anaphora_epochs
        phase = 'train'
        print('train is started...')
        for epoch_i, epoch in enumerate(ar_epochs):
            self.model.train()
            
            for i, video_id in enumerate(self.video_ids):
                
                video_visual_vectors = self.visual_vectors[video_id]['step_based_features']
                negative_video_visual_vectors = self.visual_vectors[video_id]['negative_step_based_features']

                for language in self.languages:

                    if language == 'en':
                        en_recipe = self.en_corpus.recipes[video_id]
                        en_language_vectors = self.language_vectors[video_id]['english_features']

                        sample = RecipeSample(recipe=en_recipe, 
                                              language_vectors=en_language_vectors, 
                                              visual_vectors=video_visual_vectors,
                                              is_gold=self.is_gold, 
                                              phase=phase, l=5)
                    
                    elif language == 'tr':
                        tr_recipe = self.tr_corpus.recipes[video_id]
                        tr_language_vectors = self.language_vectors[video_id]['turkish_features']

                        sample = RecipeSample(recipe=tr_recipe, 
                                              language_vectors=tr_language_vectors, 
                                              visual_vectors=video_visual_vectors,
                                              is_gold=self.is_gold, 
                                              phase=phase, l=5)

                    elif language == 'de':
                        de_recipe = self.de_corpus.recipes[video_id]
                        de_lanuage_vectors = self.language_vectors[video_id]['german_features']

                        sample = RecipeSample(recipe=de_recipe, 
                                              language_vectors=de_lanuage_vectors, 
                                              visual_vectors=video_visual_vectors,
                                              is_gold=self.is_gold, 
                                              phase=phase, l=5)

                    self.optimizer.zero_grad()
                    

                    resolution_sigmoids, \
                    resolution_probabilities, \
                    resolution_labels = self.model(sample=sample)
                    
                    resolution_labels =  resolution_labels.unsqueeze(1)
                    ar_loss = self.ar_loss_function(resolution_probabilities, resolution_labels)
                    ar_epoch_loss += ar_loss.item()
                    ar_loss.backward()
                    self.optimizer.step()
            
            break
            '''
            if epoch_i % 50 == 0 and epoch_i >= 0:
                print(f'[{epoch_i}] ar_epoch_loss: {ar_epoch_loss}')
                print()
                ar_epoch_loss = 0.0
                if epoch_i >= 250:
                    model_path = os.path.join(self.model_path+'_'+str(epoch_i)+'.pth')
                    torch.save(self.model.state_dict(), model_path)
            '''
def main():
    print()
    en_corpus = get_recipes(language='en')
    tr_corpus = get_recipes(language='tr')
    de_corpus = get_recipes(language='de')
    print()

    IS_GOLD = True
    IS_MMODAL = True
    languages = ['en']
    print()
    print('###############################')
    print('IS_MMODAL: ', IS_MMODAL)
    print('IS_GOLD: ', IS_GOLD)
    print('languages: ', languages)
    print('###############################')
    print()
    
    language_vector_path = '/netscratch/oguz/impress/mmar/data/features/language_vectors_with_step_pooler.pickle'
    with open(language_vector_path, "rb") as language_vector_file:
        language_vectors = cPickle.load(language_vector_file)

    visual_vector_path = '/netscratch/oguz/impress/mmar/data/features/negative_key_visual_vectors.pickle'
    with open(visual_vector_path, "rb") as visual_vector_file:
        visual_vectors = cPickle.load(visual_vector_file)

    print('vectors are loaded!!!')
    
    trainer = Trainer(en_corpus=en_corpus,
                      tr_corpus=tr_corpus,
                      de_corpus=de_corpus,
                      language_vectors=language_vectors,
                      visual_vectors=visual_vectors,
                      is_gold=IS_GOLD,
                      learning_rate=0.0001,
                      anaphora_epochs=451,
                      languages=languages)
    trainer.train()
    
if __name__ == '__main__':
    main()
