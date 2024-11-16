import os
import re
import json
import glob as glob
import numpy as np

def parse_entity_atts(file, string_with_reference, parenthese_inside):
	parenthese_inputs = parenthese_inside.split('||')
	entitiy_id = parenthese_inputs[0]
	antecedents = []
	relation = 'None'
	if len(parenthese_inputs) > 1:
		antecedents = parenthese_inputs[1].split(',')
		relation = parenthese_inputs[2]
	return entitiy_id, antecedents, relation


def _get_action_objects(file, line, line_number, step_id, pred, syntactic_type):
	objects = []
	strings_with_references = line.split(']:')[1].strip()
	strings_with_references = strings_with_references.split(', ')

	for string_with_reference in strings_with_references:
		string_with_reference = string_with_reference.strip()
		if re.search(r"\((.*?)\)", string_with_reference):
			parenthese_inside = re.search(r"\((.*?)\)", string_with_reference).group(1)
			parenthese_inside = parenthese_inside.strip()
			entitiy_id, antecedents, relation = parse_entity_atts(file, string_with_reference, parenthese_inside)
			string = re.sub(r"\((.*?)\)", "", string_with_reference).strip()
			if string:
				if string in ['null', 'NULL']:
					objects.append({'id':entitiy_id,
									'step_id':step_id,
									'pred':pred,
									'string':'NULL',
									'reference':antecedents,
									'relation':relation,
									'syntactic_type':syntactic_type, 
									'semantic_type':'',  
									'prep':''})
				else:
					objects.append({'id':entitiy_id,
									'step_id':step_id,
									'pred':pred,
									'string':string.lower(),
									'reference':antecedents,
									'relation':relation,
									'syntactic_type':syntactic_type, 
									'semantic_type':'',  
									'prep':''})
			else:
				objects.append({'id':entitiy_id,
								'step_id':step_id,
								'pred':pred,
								'string':'NULL',
								'reference':antecedents,
								'relation':relation,
								'syntactic_type':syntactic_type, 
								'semantic_type':'',  
								'prep':''})
	return objects

def parser(files):
	out = {}
	unique_words = set()
	for file_i, file in enumerate(files):
		f = os.path.basename(file)
		f = f.replace('.en.vtt', '')
		out[f] = {}
		with open(file, 'r') as read_file:
			lines = read_file.read().split('\n')
			step_count = -1

			for line in lines:
				if line:
					if line.startswith('0'):
						step_count += 1
						step_counter = str(step_count)
						out[f][step_counter] = {}
						out[f][step_counter]['start'] = line.split(' --> ')[0]
						out[f][step_counter]['end'] = line.split(' --> ')[1]
						out[f][step_counter]['objects'] = []
						object_line_counter = 0
					else:
						if line.startswith('ANNOT:'):
							annot = line.replace('ANNOT:', '').strip().lower()
							out[f][step_counter]['annot'] = annot
							for word in annot.split():
								unique_words.add(word)
						
						elif line.startswith('STEP_ID:'):
							out[f][step_counter]['step_id'] = 'S'+line.replace('STEP_ID:', '').strip()

						elif line.startswith('PRED:'):
							preds = line.replace('PRED:', '').strip().split(', ')
							out[f][step_counter]['pred'] = preds
						
						elif line.startswith('[OBJECTS'):
							step_id = out[f][step_counter]['step_id']
							pred = out[f][step_counter]['pred']
							syntactic_type = ''
							line_objects = _get_action_objects(file, line, object_line_counter, step_id, pred, syntactic_type)

							for obj in line_objects:
								out[f][step_counter]['objects'].append(obj)
							object_line_counter += 1
						else:
							print(file)
							print(line)
							print('There is a big problem')
							print()
				else:
					pass
	return (out, unique_words)


def get_en_recipe_data():
	en_train_mmar_files = '/netscratch/oguz/impress/mmar/md_ar/github/datasets/en_train'
	en_test_mmar_files = '/netscratch/oguz/impress/mmar/md_ar/github/datasets/en_test'
	
	out = {}
	out['train'] = {}
	train_files = [file for file in glob.glob(en_train_mmar_files+'/*', recursive = True)]

	train_out, _ = parser(train_files)
	out['train'] = train_out

	out['test'] = {}
	test_files = [file for file in glob.glob(en_test_mmar_files+'/*', recursive = True)]
	test_out, _ = parser(test_files)
	out['test'] = test_out
	out['recipes'] = {**out['train'], **out['test']}

	return out


if __name__ == '__main__':
	out = get_en_recipe_data()
