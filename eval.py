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
sys.path.insert(0, '/netscratch/oguz/impress/mmar/md_ar')
from data.head_recipes import *
from data.head_spans import *
from utils import *

from model import ResolutionModel


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


		self.test_video_files = []
		with open('../../../data/data_files/test_video_ids.txt', 'r') as en_test_video_ids_read_file:
			lines = en_test_video_ids_read_file.readlines()
			for line in lines:
				self.test_video_files.append(line.replace('\n',''))

		model = ResolutionModel()
		
		self.model = to_cuda(model)

	def load_model(self, test_model_path):
		self.model.load_state_dict(torch.load(test_model_path))
		self.model.eval()


	def ar_evaluate(self, language):
		phase = 'test'

		gold_reference_number = 0
		predicted_reference_number = 0
		correctly_predicted_reference_number = 0

		gold_reference_list = []
		predictions_reference_list = []

		gold_list = []
		prediction_list = []

		threshold = to_cuda((torch.Tensor([0.5])))
		pair_preds_golds = []
		is_null_test = 0

		with torch.no_grad():
			for i, video_id in enumerate(self.test_video_files):
				
				video_visual_vectors = self.visual_vectors[video_id]['step_based_features']
				negative_video_visual_vectors = self.visual_vectors[video_id]['negative_step_based_features']

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

				resolution_sigmoids, resolution_probabilities, resolution_labels = self.model(sample)

				for i, (anaphor_index, antecedents_labels) in enumerate(sample.anaphor2antecedent.items()):
					predictions_for_reference = []
					gold_antecedents = [i for i, e in enumerate(antecedents_labels['labels']) if e == 1]
					anaphor_span = sample.spans[anaphor_index]

					#if anaphor_span.is_null >= is_null_test:

					probability = resolution_sigmoids[i]
					max_item = torch.max(probability).item()

					if max_item == threshold[0]:
						predicted_antecedents = [torch.argmax(probability).item()]
					else:
						out = (probability > threshold).int() * 1
						inds = (out == 1).nonzero(as_tuple=True)[0]
						predicted_antecedents = inds.tolist()

					predicted_reference_number += sum(i > 0 for i in predicted_antecedents)
					gold_reference_number += sum(i > 0 for i in gold_antecedents)

					for g in gold_antecedents:
						if g > 0:
							if g in predicted_antecedents:
								predictions_for_reference.append(1)
							else:
								predictions_for_reference.append(0)

					prediction_list += predictions_for_reference

		correctly_predicted_reference_number = prediction_list.count(1)
		precision = correctly_predicted_reference_number/predicted_reference_number*100
		recall = correctly_predicted_reference_number/gold_reference_number*100
		fscore = 2 * ((precision*recall) / (precision+recall))
		print('full')
		print(language, ':', 'p \t', 'r \t', 'f1')
		# round(5.76543, 2)
		print(round(precision,2), '\t', round(recall,2), '\t', round(fscore,2))
		print()


	def ar_evaluate_zero(self, language):
		phase = 'test'

		gold_reference_number = 0
		predicted_reference_number = 0
		correctly_predicted_reference_number = 0

		gold_reference_list = []
		predictions_reference_list = []

		gold_list = []
		prediction_list = []

		threshold = to_cuda((torch.Tensor([0.5])))
		pair_preds_golds = []
		is_null_test = 0

		with torch.no_grad():
			for i, video_id in enumerate(self.test_video_files):
				
				video_visual_vectors = self.visual_vectors[video_id]['step_based_features']
				negative_video_visual_vectors = self.visual_vectors[video_id]['negative_step_based_features']

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

				resolution_sigmoids, resolution_probabilities, resolution_labels = self.model(sample)

				for i, (anaphor_index, antecedents_labels) in enumerate(sample.anaphor2antecedent.items()):
					predictions_for_reference = []
					gold_antecedents = [i for i, e in enumerate(antecedents_labels['labels']) if e == 1]
					anaphor_span = sample.spans[anaphor_index]

					if anaphor_span.is_null == 1:

						probability = resolution_sigmoids[i]
						max_item = torch.max(probability).item()

						if max_item == threshold[0]:
							predicted_antecedents = [torch.argmax(probability).item()]
						else:
							out = (probability > threshold).int() * 1
							inds = (out == 1).nonzero(as_tuple=True)[0]
							predicted_antecedents = inds.tolist()

						predicted_reference_number += sum(i > 0 for i in predicted_antecedents)
						gold_reference_number += sum(i > 0 for i in gold_antecedents)

						for g in gold_antecedents:
							if g > 0:
								if g in predicted_antecedents:
									predictions_for_reference.append(1)
								else:
									predictions_for_reference.append(0)

						prediction_list += predictions_for_reference

		correctly_predicted_reference_number = prediction_list.count(1)
		precision = correctly_predicted_reference_number/predicted_reference_number*100
		recall = correctly_predicted_reference_number/gold_reference_number*100
		fscore = 2 * ((precision*recall) / (precision+recall))
		print('zero')
		print(language, ':', 'p \t', 'r \t', 'f1')
		# round(5.76543, 2)
		print(round(precision,2), '\t', round(recall,2), '\t', round(fscore,2))
		print()


	def ar_evaluate_nominal(self, language):
		phase = 'test'

		gold_reference_number = 0
		predicted_reference_number = 0
		correctly_predicted_reference_number = 0

		gold_reference_list = []
		predictions_reference_list = []

		gold_list = []
		prediction_list = []

		threshold = to_cuda((torch.Tensor([0.5])))
		pair_preds_golds = []
		is_null_test = 0

		with torch.no_grad():
			for i, video_id in enumerate(self.test_video_files):
				
				video_visual_vectors = self.visual_vectors[video_id]['step_based_features']
				negative_video_visual_vectors = self.visual_vectors[video_id]['negative_step_based_features']

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

				resolution_sigmoids, resolution_probabilities, resolution_labels = self.model(sample)

				for i, (anaphor_index, antecedents_labels) in enumerate(sample.anaphor2antecedent.items()):
					predictions_for_reference = []
					gold_antecedents = [i for i, e in enumerate(antecedents_labels['labels']) if e == 1]
					anaphor_span = sample.spans[anaphor_index]

					if anaphor_span.is_null == 0:

						probability = resolution_sigmoids[i]
						max_item = torch.max(probability).item()

						if max_item == threshold[0]:
							predicted_antecedents = [torch.argmax(probability).item()]
						else:
							out = (probability > threshold).int() * 1
							inds = (out == 1).nonzero(as_tuple=True)[0]
							predicted_antecedents = inds.tolist()

						predicted_reference_number += sum(i > 0 for i in predicted_antecedents)
						gold_reference_number += sum(i > 0 for i in gold_antecedents)

						for g in gold_antecedents:
							if g > 0:
								if g in predicted_antecedents:
									predictions_for_reference.append(1)
								else:
									predictions_for_reference.append(0)

						prediction_list += predictions_for_reference

		correctly_predicted_reference_number = prediction_list.count(1)
		precision = correctly_predicted_reference_number/predicted_reference_number*100
		recall = correctly_predicted_reference_number/gold_reference_number*100
		fscore = 2 * ((precision*recall) / (precision+recall))
		print('nominal')
		print(language, ':', 'p \t', 'r \t', 'f1')
		# round(5.76543, 2)
		print(round(precision,2), '\t', round(recall,2), '\t', round(fscore,2))
		print()



def main():
	
	print()
	en_corpus = get_recipes(language='en')
	tr_corpus = get_recipes(language='tr')
	de_corpus = get_recipes(language='de')
	print()

	language_vector_path = '/netscratch/oguz/impress/mmar/data/features/language_vectors_with_step_pooler.pickle'
	with open(language_vector_path, "rb") as language_vector_file:
		language_vectors = cPickle.load(language_vector_file)

	visual_vector_path = '/netscratch/oguz/impress/mmar/data/features/negative_key_visual_vectors.pickle'
	with open(visual_vector_path, "rb") as visual_vector_file:
		visual_vectors = cPickle.load(visual_vector_file)

	IS_GOLD = True
	IS_MMODAL = True
	languages = ['en', 'tr', 'de']
	model_folder = '/netscratch/oguz/impress/mmar/trained_modes_md_ar/gold_small_model/wmm_cat/en_ar'
	print()
	print('###############################')
	print('IS_GOLD: ', IS_GOLD)
	print('languages: ', languages)
	print('model_folder: ', model_folder)
	print('###############################')
	print()

	trainer = Trainer(en_corpus=en_corpus,
					  tr_corpus=tr_corpus,
					  de_corpus=de_corpus,
					  language_vectors=language_vectors,
					  visual_vectors=visual_vectors,
					  is_gold=IS_GOLD,
					  learning_rate=0.0001,
					  anaphora_epochs=300,
					  languages=languages)

	model_prefix = 'gold_451_15_06_24_00_58_'
	epochs = ['300', '350', '400']

	for epoch in epochs:
		
		model_name = model_prefix + epoch + '.pth'
		print()
		print('model_name: ',model_name)
		print()

		test_model_path = os.path.join(model_folder, model_name)
		trainer.load_model(test_model_path=test_model_path)

		trainer.ar_evaluate_nominal(language='en')
		trainer.ar_evaluate_zero(language='en')
		trainer.ar_evaluate(language='en')


	print()
	print('#####################################################################')
	print('#####################################################################')
	print('#####################################################################')
	print()
	
	for epoch in epochs:
		
		model_name = model_prefix + epoch + '.pth'
		print()
		print('model_name: ',model_name)
		print()
		test_model_path = os.path.join(model_folder, model_name)
		trainer.load_model(test_model_path=test_model_path)

		trainer.ar_evaluate_nominal(language='tr')
		trainer.ar_evaluate_zero(language='tr')
		trainer.ar_evaluate(language='tr')

	print()
	print('#####################################################################')
	print('#####################################################################')
	print('#####################################################################')
	print()

	for epoch in epochs:
		
		model_name = model_prefix + epoch + '.pth'
		print()
		print('model_name: ',model_name)
		print()
		test_model_path = os.path.join(model_folder, model_name)
		trainer.load_model(test_model_path=test_model_path)

		trainer.ar_evaluate_nominal(language='de')
		trainer.ar_evaluate_zero(language='de')
		trainer.ar_evaluate(language='de')




if __name__ == '__main__':
	main()
