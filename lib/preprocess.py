#!/usr/bin/python

def remove_quotation(data):
	data_processed = []
	for i in range(0, len(data), 3):
		sentence = data[i].split()
		tag = data[i+1].split()
		NE = data[i+2].split()
		if '"' not in sentence:
			data_processed.extend(data[i:i+3])
		else:
			record = []
			for j in range(len(sentence)):
				if sentence[j] == '"':
					record.append(j)
			record = record[::-1]
			for j in record:
				sentence.pop(j)
				tag.pop(j)
				NE.pop(j)
			data_processed.extend(['\t'.join(sentence), '\t'.join(tag), '\t'.join(NE)])

	return data_processed
