#!/usr/bin/python
import sys

# from lib.preprocess import *
from lib.HMM import *
import csv

def read_data(file):
	with open(file, 'r') as f:
		data = f.readlines()
	result = data
	# result = remove_quotation(data,'"','"')
	return result

def extract(words, entities, word_NE, lexicon):
	for i in range(len(words)):
		if (words[i], entities[i]) not in lexicon.keys():
			lexicon[(words[i], entities[i])] = 1
		else:
			lexicon[(words[i], entities[i])] += 1
		word_NE[entities[i]] = word_NE[entities[i]] + 1
		if entities[i] is not 'O':
			print(entities[i], words[i], file=sys.stderr)

def run(result, bigram):
	count = 0
	for i in range(0, len(result), 3):
		if result[i] is None or len(result[i]) < 1:
			continue
		s = result[i].split()
		number = result[i+2].split()
		obrversed = bigram.viterbi(s)
		for n,o in zip(number, obrversed):
			if o is 'O':
				continue
			state[o[2:]].append(int(n))

	result = {"ORG":[], "MISC":[], "PER":[], "LOC":[]}
	for s in state.keys():
		nums = state[s]
		left = 0
		right = 0
		while right < len(nums):
			if right+1 < len(nums) and nums[right] + 1 == nums[right+1]:
				right += 1
			elif right+1 < len(nums) and nums[right] + 1 != nums[right+1]:
				result[s].append("%d-%d" % (nums[left], nums[right]))
				right += 1
				left = right
			if right+1 == len(nums):
				result[s].append("%d-%d" % (nums[left], nums[right]))
				break
		result[s] = ' '.join(result[s])
	rows = []
	headers = ['Type','Prediction']
	for key in t:
		val = result[key]
		rows.append((key, val))
	
	with open('sub_1.csv','w') as f:
		f_csv = csv.writer(f)
		f_csv.writerow(headers)
		f_csv.writerows(rows)


def cross_validation(data):
	l = len(data)
	k_list = [k*0.1 for k in range(1, 11)]
	k_list.extend([k for k in range(2, 11)])
	for (k1, k2) in itertools.product(k_list, k_list):
		print(k1, k2, file=sys.stderr)
		bigram = N_Gram(data)
		bigram.build()
		bigram.hmm(k1,k2)

		run(data, bigram)

def checkNumber(words):
	if words.isdigit():
		return True
	if words[0] is '+' or words is ':' or words is '?' or words is '!' or '*' in words:
		return True
	if words.replace(",","").isdigit() or words.replace(",","").replace(".","",1).isdigit() or words.replace(".","",1).isdigit():
		return True
	if words.find('/') is not -1:
		if words[0:words.find('/')].isdigit():
			return True
	if words.find('-') is not -1:
		if words[0:words.find('-')].isdigit():
			return True
	if words.find(':') is not -1:
		if words[0:words.find(':')].isdigit():
			return True
	return False

result = read_data('train.txt')
# cross_validation(result)

""" output """
with open('test.txt', 'r') as f:
	test = f.readlines()
bigram = N_Gram(result)
bigram.build()
bigram.hmm(0.1,0.1)
run(test, bigram)

