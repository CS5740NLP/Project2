#!/usr/bin/python
import sys

from lib.preprocess import *
from lib.HMM import *

def read_data(file):
	with open(file, 'r') as f:
		data = f.readlines()
	result = data
	# result = remove_quotation(data)
	return result
"""
def extract_old(words, entities, word_NE, count_dict):
	ner_start = False
	ne = None
	temp = []
	for i in range(len(words)):
		if not ner_start and entities[i] is 'O':
			if words[i] not in word_NE['O']:
				word_NE['O'].append(words[i])
			if words[i] not in count_dict:
				count_dict[words[i]] = 1
			else:
				count_dict[words[i]] += 1

		elif ner_start and entities[i] is 'O':
			ner_start = False

			if words[i] not in word_NE['O']:
				word_NE['O'].append(words[i])

			if '\t'.join(temp) not in word_NE[ne]:
				word_NE[ne].append('\t'.join(temp))
			if '\t'.join(temp) not in count_dict:
				count_dict['\t'.join(temp)] = 1
			else:
				count_dict['\t'.join(temp)] += 1
			
			print(ne, file=sys.stderr)
			print('\t'.join(temp), file=sys.stderr)
			# count_dict
			temp = []

		# a new NER
		elif not ner_start and entities[i][0] is 'B':
			ner_start = True
			ne = entities[i][2:]
			temp.append(words[i])

		# a new NER following another NER
		elif ner_start and entities[i][0] is 'B':

			if '\t'.join(temp) not in word_NE[ne]:
				word_NE[ne].append('\t'.join(temp))
			
			if '\t'.join(temp) not in count_dict:
				count_dict['\t'.join(temp)] = 1
			else:
				count_dict['\t'.join(temp)] += 1
			print(ne, file=sys.stderr)
			print('\t'.join(temp), file=sys.stderr)

			temp = []
			ne = entities[i][2:]
			temp.append(words[i])

		elif entities[i][0] is 'I':
			temp.append(words[i])
	
	if ner_start:
		if '\t'.join(temp) not in word_NE[ne]:
			word_NE[ne].append('\t'.join(temp))
		if '\t'.join(temp) not in count_dict:
			count_dict['\t'.join(temp)] = 1
		else:
			count_dict['\t'.join(temp)] += 1
		
		print(ne, file=sys.stderr)
		print('\t'.join(temp), file=sys.stderr)
"""
def extract(words, entities, word_NE, lexicon):
	for i in range(len(words)):
		if (words[i], entities[i]) not in lexicon.keys():
			lexicon[(words[i], entities[i])] = 1
		else:
			lexicon[(words[i], entities[i])] += 1
		word_NE[entities[i]] = word_NE[entities[i]] + 1
		if entities[i] is not 'O':
			print(entities[i], words[i], file=sys.stderr)

def cross_validation(data):
	l = len(data)
	for i in range(1,2):
		print("validation index %d." %i, file=sys.stderr)
		# cross validation
		trn_len = l//10
		tst_data = data[(i-1)*trn_len: i*trn_len]
		trn_data = data[0:(i-1)*trn_len]
		trn_data.extend(data[i*trn_len:])

		# build lexicon, just a list of (short) words of corresponding NER
		word_NE = {}
		word_NE['B-PER'] = 0
		word_NE['I-PER'] = 0
		word_NE['B-LOC'] = 0
		word_NE['I-LOC'] = 0
		word_NE['B-ORG'] = 0
		word_NE['I-ORG'] = 0
		word_NE['B-MISC'] = 0
		word_NE['I-MISC'] = 0
		word_NE['O'] = 0

		lexicon = {}

		for i in range(0, len(trn_data), 3):
			print("line: ",i, file=sys.stderr)
			print("Original: ",trn_data[i], file=sys.stderr)
			print("After...: ",trn_data[i+2], file=sys.stderr)

			words = trn_data[i].split()
			entities = trn_data[i+2].split()
			extract(words, entities, word_NE, lexicon)
			
			print("\n", file=sys.stderr)

		count = 0
		for key in word_NE.keys():
			count += word_NE[key]
		word_NE['total'] = count
		# """
		print(word_NE['B-PER'])
		print(word_NE['I-PER'])
		print(word_NE['B-ORG'])
		print(word_NE['I-ORG'])
		print(word_NE['B-LOC'])
		print(word_NE['I-LOC'])
		print(word_NE['B-MISC'])
		print(word_NE['I-MISC'])
		print(word_NE['O'])
		print(word_NE['total'])
		# """

		# for key, val in lexicon.items():
		# 	print(key, val)		

		word_NE_prb = {}

		for key in lexicon.keys():
			word = key[0]
			ne = key[1]
			word_NE_prb[(word, ne)] = lexicon[key]/word_NE[ne]
		for key, val in word_NE_prb.items():
			print(key, val)	

result = read_data('train.txt')

bigram = N_Gram(result)
bigram.build()
bigram.hmm()
# bigram.get_dict()

count = 0
for i in range(0, len(result), 3):
	s = result[i].split()
	ne = result[i+2].split()
	obrversed = bigram.viterbi(s)
	check = (obrversed==ne)
	if not check:
		count += 1
	print(check, obrversed)

print("precision:", count*3/len(result))


