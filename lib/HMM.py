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
		# self.final_probability = {'B-PER':0, 'I-PER':0,'B-LOC':0,'I-LOC':0,'B-ORG':0,'I-ORG':0,'B-MISC':0,'I-MISC':0, 'O':0}
		self.transition_probablity = {'B-PER':{}, 'I-PER':{},'B-LOC':{},'I-LOC':{},'B-ORG':{},'I-ORG':{},'B-MISC':{},'I-MISC':{}, 'O':{}}
		self.emission_probability = {'B-PER':{}, 'I-PER':{},'B-LOC':{},'I-LOC':{},'B-ORG':{},'I-ORG':{},'B-MISC':{},'I-MISC':{}, 'O':{}}
		
		self.states = ['B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC', 'O']
		# self.observation = data

	def build(self):
		for i in range(0, len(self.data), 3):
			entities = self.data[i+2].split()
			words = self.data[i].split()
			for j in range(len(entities)):
				# start
				if words[j].lower() not in self.emission_probability[entities[j]].keys():
					self.emission_probability[entities[j]][words[j].lower()] = 1
				elif words[j].lower() in self.emission_probability[entities[j]].keys():
					self.emission_probability[entities[j]][words[j].lower()] = self.emission_probability[entities[j]][words[j].lower()] + 1

				if j is 0 and (None, entities[j]) not in self.gram_dict.keys():
					self.gram_dict[(None, entities[j])] = 1

				elif j is not 0 and j < len(entities) and (entities[j-1], entities[j]) not in self.gram_dict.keys():
					self.gram_dict[(entities[j-1], entities[j])] = 1
				# end 
				# elif (entities[j], None) not in self.gram_dict.keys():
				# 	self.gram_dict[(entities[j], None)] = 1

				# start 
				elif j is 0 and (None, entities[j]) in self.gram_dict.keys():
					self.gram_dict[(None, entities[j])] = self.gram_dict[(None, entities[j])] + 1
				elif j is not 0 and j < len(entities) and (entities[j-1], entities[j]) in self.gram_dict.keys():
					self.gram_dict[(entities[j-1], entities[j])] = self.gram_dict[(entities[j-1], entities[j])] + 1
				# end 
				# elif (entities[j], None) in self.gram_dict.keys():
				# 	self.gram_dict[(entities[j], None)] = self.gram_dict[(entities[j], None)] + 1
				
				self.NER_dict[entities[j]] = self.NER_dict[entities[j]] + 1
				self.NER_total += 1
	
	def hmm(self):
		for key in self.states:
			if (None, key) in self.gram_dict.keys():
				self.start_probability[key] = self.gram_dict[(None, key)]/self.NER_total
			# elif (key, None) in self.gram_dict.keys():
			# 	self.final_probability[key] = self.gram_dict[(key, None)]/self.NER_total
	
		for key, val in self.gram_dict.items():
			if key[0] is None or key[1] is None:
				continue
			# key[0] --> key[1]
			self.transition_probablity[key[0]][key[1]] = val / self.NER_dict[key[0]]
		
		for (k0, k1) in itertools.product(self.states, self.states):
			if k1 in self.transition_probablity[k0]:
				continue
			self.transition_probablity[k0][k1] = 0

		for entity, words in self.emission_probability.items():
			for key, val in words.items():
				self.emission_probability[entity][key] = val / self.NER_dict[entity]

	def viterbi(self, ob_sequence):
		V = {}
		BPTR = { s:[] for s in self.states }

		for state in self.states: # intialization
			if ob_sequence[0].lower() in self.emission_probability[state].keys():
				V[state] = self.start_probability[state] * self.emission_probability[state][ob_sequence[0].lower()]
			else:
				V[state] = 0

		for t in range(1, len(ob_sequence)):
			prev_V = V
			V = {}
			for s in self.states:
				if ob_sequence[t].lower() in self.emission_probability[s]:
					V[s], prev_state = max([(prev_V[k] * self.transition_probablity[k][s] * self.emission_probability[s][ob_sequence[t].lower()], k) for k in self.states])
					BPTR[s].append(prev_state)
				else:
					V[s] = 0
					BPTR[s].append('UNK')

		max_pro = -1
		max_path = None
		for s in self.states:
			BPTR[s].append(s)
			if V[s] > max_pro:
				max_path = BPTR[s]
				max_pro = V[s]

		return max_path

	# print out function
	def get_dict(self):
		for key, val in self.emission_probability.items():
			print(key, val, file=sys.stderr)

		
