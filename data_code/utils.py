import json
import csv
import os
import math
import re
import shutil
import collections
import glob as glob
import numpy as np

from boltons.iterutils import pairwise, windowed
from itertools import groupby, combinations
from collections import defaultdict
import os, io, re, attr, random
import _pickle as cPickle


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence

# pair = anaphor.index, antecedent.index, anaphor.step_index-antecedent.step_index
def analysis(sppg):
	for spans, pair, prediction, gold in sppg:
		print(pair)
		anaphor_index = pair[0][0]
		if 0 not in gold:
			print(gold)
			print(prediction)
			print(spans[anaphor_index].string)
			print('Gold:')
			for g in gold: 
				print(spans[g].step_index, spans[g].string)

			print('Prediction:')
			for p in prediction: 
				print(spans[p].step_index, spans[p].string)
			print()
			print()

def to_cuda(x):
	""" GPU-enable a tensor """
	if torch.cuda.is_available():
		x = x.cuda()
	return x

def to_var(x):
	""" Convert a tensor to a backprop tensor and put on GPU """
	return to_cuda(x).requires_grad_()

def unpack_and_unpad(lstm_out, reorder):
	""" Given a padded and packed sequence and its reordering indexes,
	unpack and unpad it. Inverse of pad_and_pack """

	# Restore a packed sequence to its padded version
	unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

	# Restored a packed sequence to its original, unequal sized tensors
	unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

	# Restore original ordering
	regrouped = [unpadded[idx] for idx in reorder]

	return regrouped

def pad_and_stack(tensors, pad_size=None, value=0):
	""" Pad and stack an uneven tensor of token lookup ids.
	Assumes num_sents in first dimension (batch_first=True)"""

	# Get their original sizes (measured in number of tokens)
	sizes = [s.shape[0] for s in tensors]

	# Pad size will be the max of the sizes
	if not pad_size:
		pad_size = max(sizes)

	# Pad all sentences to the max observed size
	# TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
	padded = torch.stack([F.pad(input=sent[:pad_size],
								pad=(0, 0, 0, max(0, pad_size-size)),
								value=value)
						  for sent, size in zip(tensors, sizes)], dim=0)

	return padded, sizes

def pad_and_stack_labels(tensors, pad_size=None, value=0):
	""" Pad and stack an uneven tensor of token lookup ids.
	Assumes num_sents in first dimension (batch_first=True)"""

	# Get their original sizes (measured in number of tokens)
	sizes = [s.shape[0] for s in tensors]

	# Pad size will be the max of the sizes
	if not pad_size:
		pad_size = max(sizes)

	# Pad all sentences to the max observed size
	# TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
	padded = torch.stack([F.pad(input=sent[:pad_size], pad=(0, 0, 0, max(0, pad_size-size)), value=0)
						  for sent, size in zip(tensors, sizes)], dim=0)

	return padded

def pack(tensors):
	""" Pack list of tensors, provide reorder indexes """

	# Get sizes
	sizes = [t.shape[0] for t in tensors]

	# Get indexes for sorted sizes (largest to smallest)
	size_sort = np.argsort(sizes)[::-1]

	# Resort the tensor accordingly
	sorted_tensors = [tensors[i] for i in size_sort]

	# Resort sizes in descending order
	sizes = sorted(sizes, reverse=True)

	# Pack the padded sequences
	packed = pack_sequence(sorted_tensors)

	# Regroup indexes for restoring tensor to its original order
	reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)

	return packed, reorder


def find_index_of_sublist(sl,l):
	results=[]
	sll=len(sl)
	for ind in (i for i,e in enumerate(l) if e==sl[0]):
		if l[ind:ind+sll]==sl:
			results.append(ind)
			results.append(ind+sll-1)
	return results



class TestVector(object):
	"""docstring for Vectors"""
	def __init__(self, test_corpus, region_dictionary_path, bert_vector_folder, clip_text_folder, clip_bbox_folder):
		super(TestVector, self).__init__()
		self.bert_layer = 2
		self.region_vector = '/netscratch/oguz/impress/odar_data/test_bbox_annotations/bbox_images_annotations/test_regions'
		self.test_corpus = test_corpus
		self.video2regions = region_dictionary_path
		self.bert_vector_folder = bert_vector_folder
		self.clip_text_folder = clip_text_folder
		self.clip_bbox_folder = clip_bbox_folder
		self.recipe2vectors = self.load_vectors()

	def load_vectors(self):
		out = {}

		video2regions = {}
		with open(self.video2regions) as video_region_file:
			video2regions = json.load(video_region_file)

		for i, recipe in enumerate(self.test_corpus.recipes):
			print(recipe.video_id)
			out[recipe.video_id] = {}
			video_instruction_folder = os.path.join(self.clip_text_folder, recipe.video_id)
			
			recipe_bert_vector_path = os.path.join(self.bert_vector_folder, recipe.video_id+'.npy')
			with open(recipe_bert_vector_path, 'rb') as bert_file:
				bert_vector = np.load(bert_file)
			bert_vector = bert_vector[:,self.bert_layer,:]
			out[recipe.video_id]['bert_recipe_vector'] = bert_vector

			text_clip_vector_list = []
			out[recipe.video_id]['regions'] = {}
			for step_index, step in recipe.parsed_recipe.items():
				video_step_instruction_vector_path = os.path.join(video_instruction_folder, step['step_id']+'.npy')
				with open(video_step_instruction_vector_path, 'rb') as instruction_clip_file:
					instruction_clip_vector = np.load(instruction_clip_file)
				text_clip_vector_list += instruction_clip_vector.tolist()

				out[recipe.video_id]['regions'][step['step_id']] = {}
				if step['step_id'] in video2regions[recipe.video_id]:
					region_values = video2regions[recipe.video_id][step['step_id']]
					for k, step_values in region_values.items():
						item = k.replace('-','')
						out[recipe.video_id]['regions'][step['step_id']][k] = []
						for image_values in step_values:
							image_name, x, y, w, h = image_values[0], image_values[1], image_values[2], image_values[3], image_values[4]
							region_path = image_name.replace('jpg','pth')
							bbox_clip_vectors = torch.load(os.path.join(self.region_vector, region_path))
							bbox_clip_vectors = bbox_clip_vectors['feats'][:20, :]
							out[recipe.video_id]['regions'][step['step_id']][k].append((bbox_clip_vectors,image_name, x, y, w, h))
				#else:
				#	print('###############################################',recipe.video_id, step['step_id'], len(step['objects']))
				
			out[recipe.video_id]['instruction_clip_vector'] = np.stack(text_clip_vector_list)

		with open(r"../vectors/reduced_test_vectors.pickle", "wb") as output_file:
			cPickle.dump(out, output_file)
		
		return out


#train_corpus, bert_vector_folder, clip_text_folder, clip_bbox_folder
class TrainVector(object):
	"""docstring for Vectors"""
	def __init__(self, train_corpus, bert_vector_folder, clip_text_folder, clip_bbox_folder, positive_negative_file):
		super(TrainVector, self).__init__()
		self.bert_layer = 2
		self.region_vector = '/netscratch/oguz/impress/odar_data/test_bbox_annotations/bbox_images_annotations/test_regions'
		self.train_corpus = train_corpus
		self.bert_vector_folder = bert_vector_folder
		self.clip_text_folder = clip_text_folder
		self.clip_bbox_folder = clip_bbox_folder
		self.positive_negative_path = positive_negative_file
		self.recipe2vectors = self.load_vectors()

	def load_vectors(self):
		out = {}

		positive2negative = {}
		with open(self.positive_negative_path) as no_object_file:
			positive2negative = json.load(no_object_file)

		for i, recipe in enumerate(self.train_corpus.recipes):
			print(recipe.video_id)
			out[recipe.video_id] = {}

			video_instruction_folder = os.path.join(self.clip_text_folder, recipe.video_id)

			out[recipe.video_id]['negative_videos'] = []
			video_negative_videos = positive2negative[recipe.video_id]
			negative_videos = random.sample(video_negative_videos, 5)
			for negative_video in negative_videos:
				negative_video_frame_folder = os.path.join(self.clip_bbox_folder, negative_video)
				step_bboxes_frame_paths = glob.glob(negative_video_frame_folder+"/*")
				for step_bboxes_frame_path in step_bboxes_frame_paths:
					step_bboxes_frame_paths = glob.glob(step_bboxes_frame_path+"/*")
					for step_bboxes_frame_path in step_bboxes_frame_paths:
						bbox_clip_vectors = torch.load(step_bboxes_frame_path)
						bbox_clip_vectors = bbox_clip_vectors['feats'][:20, :]
						out[recipe.video_id]['negative_videos'].append(bbox_clip_vectors)
			
			recipe_bert_vector_path = os.path.join(self.bert_vector_folder, recipe.video_id+'.npy')
			with open(recipe_bert_vector_path, 'rb') as bert_file:
				bert_vector = np.load(bert_file)
			bert_vector = bert_vector[:,self.bert_layer,:]
			out[recipe.video_id]['bert_recipe_vector'] = bert_vector

			text_clip_vector_list = []
			out[recipe.video_id]['regions'] = {}

			video_bbox_folder = os.path.join(self.clip_bbox_folder, recipe.video_id)
			for step_index, step in recipe.parsed_recipe.items():
				
				video_step_instruction_vector_path = os.path.join(video_instruction_folder, step['step_id']+'.npy')
				with open(video_step_instruction_vector_path, 'rb') as instruction_clip_file:
					instruction_clip_vector = np.load(instruction_clip_file)
				text_clip_vector_list += instruction_clip_vector.tolist()

				step_bbox_path = os.path.join(video_bbox_folder, step['step_id'])
				step_bbox_file_paths = glob.glob(step_bbox_path+'/*')
				out[recipe.video_id]['regions'][step['step_id']] = []
				for step_bbox_file_path in step_bbox_file_paths:
					bbox_clip_vectors = torch.load(step_bbox_file_path)
					bbox_clip_vectors = bbox_clip_vectors['feats'][:20, :]
					out[recipe.video_id]['regions'][step['step_id']].append(bbox_clip_vectors)

			out[recipe.video_id]['instruction_clip_vector'] = np.stack(text_clip_vector_list)
		with open(r"/netscratch/oguz/impress/joint_anaphora/vectors/test_vectors_2.pickle", "wb") as output_file:
			cPickle.dump(out, output_file)
		
		
		return out
