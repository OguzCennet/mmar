from boltons.iterutils import pairwise, windowed
from itertools import groupby, combinations
import os, io, re, attr, random
import numpy as np
from .utils import *
from .recipes import *

def flatten(alist):
	""" Flatten a list of lists into one list """
	out = []
	for sublist in alist:
		for item in sublist:
			out.append(item)
	return out

def _print_recipe_spans(recipe_sample):
	for span in recipe_sample.spans:
		print('------------------------------------------------------------------------')
		print('------------------------- CANDIDATE ANTECEDENTS ------------------------')
		print('------------------------------------------------------------------------')
		print('label', '\t    ', 'is_null', '\t    ', 'span_id    ', '\t', 'step_id    ', '\t', 'string')
		for y, l in zip(span.yi_spans, span.yi_labels):
			print(l, '\t\t', y.is_null, '\t\t', y.span_id, '\t\t', y.step_id, '\t\t', y.string)
		print('------------------------------------------------------------------------')
		print('---------------------------------- ANAPHOR -----------------------------')
		print('------------------------------------------------------------------------')
		print('is_null', '\t:', span.is_null)
		print('span_id', '\t:', span.span_id)
		print('string', '\t\t:', span.string)
		print('antecedents', '\t:', span.antecedents)
		print()
		print()
		

@attr.s(frozen=True, repr=False)
class ModelSpan:
	# Id within considered batch for traning (candidate, or gold)
	index = attr.ib()
	# Step number
	step_index = attr.ib()
	# Left token indexes
	i1 = attr.ib()
	# Right token indexes
	i2 = attr.ib()
	# span string
	string = attr.ib()

	# Gold ids for spans only
	span_id = attr.ib()
	# Step id
	step_id = attr.ib()
	# is it a anaphor mention
	is_mention = attr.ib()
	# is it a antecedent mention
	is_step = attr.ib()
	# is it a antecedent mention
	is_null = attr.ib()
	# Antecedent ids
	antecedents = attr.ib()
	# relation
	relation = attr.ib()

	yi_spans = attr.ib(default=None)
	yi_ref_labels = attr.ib(default=None)
	yi_rel_labels = attr.ib(default=None)

	#span_mdeberta_vector= attr.ib(default=None)
	span_xlm_roberta_vector = attr.ib(default=None)
	#span_mclip_vector = attr.ib(default=None)
	span_clip_vectors = attr.ib(default=None)

	# Unary mention score, as tensor
	si = attr.ib(default=None)

	yi_idx = attr.ib(default=None)

	def __len__(self):
		return self.i2-self.i1+1

	def __repr__(self):
		return 'Span representing %d tokens' % (self.__len__())


@attr.s(frozen=True, repr=False)
class Span:
	# Id within total spans (for indexing into a batch computation)
	index = attr.ib()
	# Step number
	step_index = attr.ib()
	# Step id
	step_id = attr.ib()
	# Left token indexes in the full recipe
	i1 = attr.ib()
	# Right token indexes in the full recipe
	i2 = attr.ib()
	# span string
	string = attr.ib()

	# Gold ids for spans only
	span_id = attr.ib(default=None)
	# is it a anaphor mention
	is_mention = attr.ib(default=None)
	# is it a antecedent mention
	is_step = attr.ib(default=None)
	# is it a antecedent mention
	is_null = attr.ib(default=None)
	# Antecedent ids
	antecedents = attr.ib(default=None)
	# relation
	relation = attr.ib(default=None)

	def __len__(self):
		return self.i2-self.i1+1

	def __repr__(self):
		return 'Span representing %d tokens' % (self.__len__())


def compute_idx_spans_with_step_number(recipe, L=5):
	""" Compute span indexes for all possible spans up to length L in each sentence """
	idx_spans, text_spans, shift = [], [], 0
	document_tokens = []
	for ind, (sent, step_id) in enumerate(zip(recipe.steps, recipe.step_ids)):
		
		# Returns tuples with exactly length size. 
		# If the iterable is too short to make a window of length size, no tuples are returned. See windowed_iter() for more
		# >>> list(windowed_iter(range(7), 3))
		# [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
		span_indexes = flatten([windowed(range(shift, len(sent)+shift), length) for length in range(1, L)])
		sentence_indexes = list(range(shift, len(sent)+shift))
		if tuple(sentence_indexes) not in span_indexes:
			span_indexes.append(tuple(sentence_indexes))
		idx_spans.append((ind+1, span_indexes, step_id))
		shift += len(sent)
		document_tokens += sent
	return idx_spans


class RecipeSample(object):
	"""docstring for Data"""
	def __init__(self, recipe, language_vectors, visual_vectors, is_gold, phase, l):
		super(RecipeSample).__init__()
		self.label_set = {'None':0, 'ID1':1, 'NI':2,  'BR1':3,  'BR2':3,  'BR3':3, 'BR4':3}
		self.recipe = recipe
		self.language_vectors = language_vectors
		self.visual_vectors = visual_vectors
		self.is_gold = is_gold
		self.phase = phase
		self.l = l
		self.spans = self.extract_modeling_spans()
		self.recipe_pairs, self.antecedent_lenghts, self.anaphor2antecedent = self.extract_pairs()

	def extract_spans(self):
		recipe_spans = compute_idx_spans_with_step_number(self.recipe, L=self.l)
		first_spans = []
		span_start_end = []
		
		span_counter = 0
		span = Span(index=span_counter, step_index=int(0), step_id='S', i1=-100, i2=-100, string='dummy')
		first_spans.append(span)
		
		for step_index, sent_spans, step_id in recipe_spans:
			for span_indexes in sent_spans:
				span_counter += 1
				span_start = span_indexes[0]
				span_end = span_indexes[-1]
				span_string = ' '.join(self.recipe.tokens[span_start:span_end + 1])
				span = Span(index=span_counter, step_index=int(step_index), step_id=step_id, i1=span_start, i2=span_end, string=span_string)
				
				first_spans.append(span)
				span_start_end.append((span_start, span_end))
		
		second_spans = []
		for span in first_spans:
			if span.index == 0:
				second_spans.append(attr.evolve(span, span_id='E', is_mention=int(1), is_step=int(0), 
												is_null=int(0), antecedents=[], relation='None'))
			else:
				if (span.i1, span.i2) in self.recipe.mention_start_end:
					span_index = self.recipe.mention_start_end.index((span.i1, span.i2))
					mention = self.recipe.mentions[span_index]
					
					span_id = mention['id']
					step_id = mention['step_id']

					#print(mention['is_null'], step_id, span.string)
					is_step = int(mention['is_step'])
					is_null = int(mention['is_null'])
					antecedents = mention['antecedents']
					relation = mention['relation']
					
					second_spans.append(attr.evolve(span, span_id=span_id, is_mention=1, is_step=is_step,
													is_null=is_null, antecedents=antecedents, relation=relation))
				else:
					span_id = ''
					step_id = ''
					is_step = 0
					is_null = 0
					antecedents = []
					relation = 'None'
					second_spans.append(attr.evolve(span, span_id=span_id, is_mention=0, is_step=is_step, 
												    is_null=is_null, antecedents=antecedents, relation=relation))
		return second_spans

	def extract_modeling_spans(self):
		third_spans = []
		span_list = self.extract_spans()

		candidate_spans = []
		candidate_counter = 0
		
		gold_spans = []
		gold_counter = 0
		for i_span in span_list:
			candidate_spans.append(ModelSpan(index=candidate_counter, 
											 step_index=i_span.step_index, 
											 i1=i_span.i1, i2=i_span.i2, 
											 string=i_span.string,
											 span_id=i_span.span_id, 
											 step_id=i_span.step_id, 
											 is_mention=i_span.is_mention, 
											 is_step=i_span.is_step, 
											 is_null=i_span.is_null, 
											 antecedents=i_span.antecedents, 
											 relation=i_span.relation))
			candidate_counter += 1



			if i_span.is_mention == 1:
				gold_spans.append(ModelSpan(index=gold_counter, 
											 step_index=i_span.step_index, 
											 i1=i_span.i1, i2=i_span.i2, 
											 string=i_span.string,
											 span_id=i_span.span_id, 
											 step_id=i_span.step_id, 
											 is_mention=i_span.is_mention, 
											 is_step=i_span.is_step, 
											 is_null=i_span.is_null, 
											 antecedents=i_span.antecedents, 
											 relation=i_span.relation))
				gold_counter += 1

		if self.is_gold:
			given_spans = gold_spans
		else:
			given_spans = candidate_spans

		self.mention_labels = []
		for m in given_spans:
			is_mention_bool = int(m.is_mention)
			is_step_bool = int(m.is_step)
			is_null_bool = int(m.is_null)
			if is_step_bool == 1:
				is_mention_bool = 0
			if is_null_bool == 1:
				is_mention_bool = 1
			if m.string == 'dummy':
				is_mention_bool = 0
			#print(is_mention_bool, m.string)
			self.mention_labels.append(is_mention_bool)

		for i, i_span in enumerate(given_spans):
			#print(i_span.string)
			yi_spans = []
			yi_ref_labels = []
			yi_rel_labels = []
			if i_span.step_index > 1:
				for j_span in given_spans[:i]:
					if j_span.step_id != i_span.step_id:
						yi_spans.append(j_span)
						if len(i_span.antecedents) == 0 and j_span.index == 0:
							yi_ref_labels.append(1)
							yi_rel_labels.append(self.label_set['None'])
						else:
							if j_span.span_id in i_span.antecedents:
								yi_ref_labels.append(1)
								yi_rel_labels.append(self.label_set[i_span.relation])
							else:
								yi_ref_labels.append(0)
								yi_rel_labels.append(self.label_set['None'])
			third_spans.append(attr.evolve(i_span, yi_spans=yi_spans, yi_ref_labels=yi_ref_labels, yi_rel_labels=yi_rel_labels))
		
		return third_spans


	def extract_pairs(self):
		recipe_pairs = []

		antecedent_lenghts = []
		anaphor2antecedent = {}
		for anaphor in self.spans:
			if anaphor.is_step != 1:
				if anaphor.step_index > 1:
					anaphor2antecedent[anaphor.index] = {}
					anaphor2antecedent[anaphor.index]['antecedents'] = []
					anaphor2antecedent[anaphor.index]['labels'] = []
					antecedents, labels = [], []

					for antecedent, label in zip(anaphor.yi_spans, anaphor.yi_ref_labels):
						if 1 not in anaphor.yi_ref_labels and antecedent.index == 0:
							recipe_pairs.append((anaphor.index, 0, anaphor.step_index-0, 1))
							antecedents.append(0)
							labels.append(1)
						else:
							recipe_pairs.append((anaphor.index, antecedent.index, anaphor.step_index-antecedent.step_index, label))
							antecedents.append(antecedent.index)
							labels.append(label)

					anaphor2antecedent[anaphor.index]['antecedents'] = antecedents
					anaphor2antecedent[anaphor.index]['lenghts'] = len(antecedents)
					anaphor2antecedent[anaphor.index]['labels'] = labels


		for anaphor_index, antecedents_labels in  anaphor2antecedent.items():
			antecedent_lenghts.append(len(antecedents_labels['antecedents']))
		return recipe_pairs, antecedent_lenghts, anaphor2antecedent


def get_spans(corpus, data_root, bert_vector_folder, clip_text_folder, clip_image_folder, is_gold):
	corpus_recipe_samples = []
	for i, recipe in enumerate(corpus.recipes):
		recipe_sample = RecipeSample(data_root=data_root, bert_vector_folder=bert_vector_folder,
									 clip_text_folder=clip_text_folder, clip_image_folder=clip_image_folder,
									 recipe=recipe, is_gold=is_gold)
		corpus_recipe_samples.append(recipe_sample)
	return corpus_recipe_samples

