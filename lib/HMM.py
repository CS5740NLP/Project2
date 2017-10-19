#!/usr/bin/python
import sys
import itertools

class N_Gram:
	"""
	docstring for HMM
	Bi-gram
	
	"""
	def __init__(self, data, n=2):
		self.n = n
		
		self.data = data
		self.gram_dict = {}
		
		self.NER_dict = {'B-PER':0, 'I-PER':0,'B-LOC':0,'I-LOC':0,'B-ORG':0,'I-ORG':0,'B-MISC':0,'I-MISC':0, 'O':0}
		self.NER_total = 0

		self.start_probability = {'B-PER':0, 'I-PER':0,'B-LOC':0,'I-LOC':0,'B-ORG':0,'I-ORG':0,'B-MISC':0,'I-MISC':0, 'O':0}
		self.transition_probability = {'B-PER':{}, 'I-PER':{},'B-LOC':{},'I-LOC':{},'B-ORG':{},'I-ORG':{},'B-MISC':{},'I-MISC':{}, 'O':{}}
		self.emission_probability = {'B-PER':{}, 'I-PER':{},'B-LOC':{},'I-LOC':{},'B-ORG':{},'I-ORG':{},'B-MISC':{},'I-MISC':{}, 'O':{}}
		
		self.states = ['B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC', 'O']

		self.word_dict = {}

	def add_k(self, k):
		for entity, words in self.emission_probability.items():
			for key, val in words.items():	
				self.emission_probability[entity][key] = (val + k)/ (self.NER_dict[entity] + k*len(self.word_dict.keys()))

	def classify_digit(self, j, digits, state, entities):
		if digits not in self.emission_probability[state].keys():
			self.emission_probability[state][digits] = 1
		elif digits in self.emission_probability[state].keys():
			self.emission_probability[state][digits] = self.emission_probability[state][digits] + 1

		if j is 0 and (None, state) not in self.gram_dict.keys():
			self.gram_dict[(None, state)] = 1
		elif j is not 0 and j < len(entities) and (entities[j-1], state) not in self.gram_dict.keys():
			self.gram_dict[(entities[j-1], state)] = 1
		elif j is 0 and (None, state) in self.gram_dict.keys():
			self.gram_dict[(None, state)] = self.gram_dict[(None, state)] + 1
		elif j is not 0 and j < len(entities) and (entities[j-1], state) in self.gram_dict.keys():
			self.gram_dict[(entities[j-1], state)] = self.gram_dict[(entities[j-1], state)] + 1

		self.NER_dict[state] = self.NER_dict[state] + 1
		self.NER_total += 1

	def build(self):
		for i in range(0, len(self.data), 3):
			entities = self.data[i+2].split()
			words = self.data[i].split()
			for j in range(len(entities)):
				if self.checkNumber(words[j]):
					self.classify_digit(j, '@', 'O', entities)
					if '@' not in self.word_dict.keys():
						self.word_dict['@'] = 1
					else:
						self.word_dict['@'] = self.word_dict['@'] + 1
					continue
				# start
				if words[j].lower() not in self.emission_probability[entities[j]].keys():
					self.emission_probability[entities[j]][words[j].lower()] = 1
					self.word_dict[words[j].lower()] = 1
				elif words[j].lower() in self.emission_probability[entities[j]].keys():
					self.emission_probability[entities[j]][words[j].lower()] = self.emission_probability[entities[j]][words[j].lower()] + 1
					self.word_dict[words[j].lower()] = self.word_dict[words[j].lower()] + 1
				
				if j is 0 and (None, entities[j]) not in self.gram_dict.keys():
					self.gram_dict[(None, entities[j])] = 1

				elif j is not 0 and j < len(entities) and (entities[j-1], entities[j]) not in self.gram_dict.keys():
					self.gram_dict[(entities[j-1], entities[j])] = 1
				# start 
				elif j is 0 and (None, entities[j]) in self.gram_dict.keys():
					self.gram_dict[(None, entities[j])] = self.gram_dict[(None, entities[j])] + 1
				elif j is not 0 and j < len(entities) and (entities[j-1], entities[j]) in self.gram_dict.keys():
					self.gram_dict[(entities[j-1], entities[j])] = self.gram_dict[(entities[j-1], entities[j])] + 1
				
				self.NER_dict[entities[j]] = self.NER_dict[entities[j]] + 1
				self.NER_total += 1
	
	def hmm(self, t1, t2):
		for key in self.states:
			if (None, key) in self.gram_dict.keys():
				self.start_probability[key] = self.gram_dict[(None, key)]*3./len(self.data)
	
		for key, val in self.gram_dict.items():
			if key[0] is None :#or key[1] is None:
				continue
			# key[0] --> key[1]
			self.transition_probability[key[0]][key[1]] = (val + t1)/(self.NER_dict[key[0]] + t1*9)
		
		for (k0, k1) in itertools.product(self.states, self.states):
			if k1 in self.transition_probability[k0]:
				continue
			if 'B-' in k1:
				self.transition_probability[k0][k1] = 1e-6
			else:
				self.transition_probability[k0][k1] = 0

		for entity, words in self.emission_probability.items():
			for key, val in words.items():	
				self.emission_probability[entity][key] = (val + t2)/ (self.NER_dict[entity] + t2*len(self.word_dict.keys()))


	def validate_path(self,path):
		if len(path) < 2:
			return path
		prev = path[0]
		for i in range(1, len(path)):
			if (prev is 'O' or prev is 'UNK') and 'I-' in path[i]:
				path[i-1] = 'B-' + path[i][2:]
			elif prev is 'UNK':
				path[i-1] = 'O'
			prev = path[i]
		# path[-1] = 'O'
		return path


	def checkNumber(self, words):
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

	def viterbi(self, ob_sequence):
		V = {}
		BPTR = { s:[] for s in self.states }

		for state in self.states: # intialization
			if self.checkNumber(ob_sequence[0]) :
				ob_sequence[0] = '@'

			if ob_sequence[0].lower() in self.emission_probability[state].keys():
				V[state] = self.start_probability[state] * self.emission_probability[state][ob_sequence[0].lower()]
			else:
				V[state] = 1e-6

		for t in range(1, len(ob_sequence)):
			prev_V = V
			V = {}
			for s in self.states:
				if self.checkNumber(ob_sequence[t]):
					ob_sequence[t] = '@'

				if ob_sequence[t].lower() in self.emission_probability[s]:
					V[s], prev_state = max([(prev_V[k] * self.transition_probability[k][s] * self.emission_probability[s][ob_sequence[t].lower()], k) for k in self.states])
					BPTR[s].append(prev_state)
				else:
					V[s] = 1e-6
					BPTR[s].append("UNK")


		max_pro = -1
		max_path = None
		for s in self.states:
			BPTR[s].append(s)
			if V[s] > max_pro:
				max_path = BPTR[s]
				max_pro = V[s]

		max_path = self.validate_path(max_path)
		return max_path

	# print out function
	def get_dict(self):
		# for key, val in self.emission_probability.items():
		# 	val = sorted(val)
		
		for v in self.emission_probability['B-PER']:
			print(self.emission_probability['B-PER'][v], v, file=sys.stderr)
				
