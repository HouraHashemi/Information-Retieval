from __future__ import unicode_literals

import pandas as pd
import math

import matplotlib.pyplot as plt
import numpy as np

import re
import time
import json
import ast
import os


import copy
import itertools
import collections


# from parsivar import Normalizer
# from parsivar import Tokenizer
# from parsivar import FindStems
# from parsivar import POSTagger

# from parsivar import FindChunks
# from parsivar import DependencyParser
# from parsivar import SpellCheck
# from hazm import *


class Browser:


	db = dict()
	inverted_index = dict()

	def __init__(self, json_path):
		
		for path in json_path:
			start_time = time.time()
			with open(path) as json_file:
				data = json.load(json_file)
			# {docid: {'title', 'content', 'tags', 'date', 'url', 'category'}}
			self.db.update(data)
			print("--- {} seconds ---".format(time.time() - start_time))
		print(len(self.db.keys()))

	
	# def tokenizer(self, content):

	# 	SYMBOLS = list('[!@#$%^&*().[]_+=-/\'\":;<>}{|`~]؟؛«»?,0123456789')
	# 	ALPHABET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

	# 	# -- normalization
	# 	normalizer = Normalizer(statistical_space_correction=True)
	# 	content = normalizer.normalize(content)

	# 	# -- tokenization
	# 	tokenizer = Tokenizer()		
	# 	words = tokenizer.tokenize_words(content)

	# 	# -- stemming
	# 	stemmer = FindStems()

	# 	# # -- tagger
	# 	tagger = POSTagger(tagging_model="stanford")
	# 	text_tags = tagger.parse(words)
	# 	# tx = dict(sorted(text_tags, key=lambda item: item[0]))
		

	# 	tokens = list()

	# 	for pair in copy.deepcopy(text_tags):
	# 		wrd = pair[0]
	# 		tag = pair[1]
	# 		if (tag in ["PO","DELM","FW","CON","PRO","DET","."]) or (len(wrd)<=2):
	# 			pass
	# 		else:
	# 			stem = stemmer.convert_to_stem(wrd)
	# 			stem = stem.split("&")
	# 			tokens = tokens + stem

	# 	return tokens, words

		
	# def create_inverted_index(self):
	# 	start_time = time.time()
	# 	stemmer = FindStems()

	# 	for docid in self.db:
	# 		print(docid)
	# 		content = self.db[docid]['content']
	# 		tokens, words = self.tokenizer(content)

	# 		for t in set(tokens):	
	# 			if t not in self.inverted_index:
	# 				# -- { token:{docid:positions} } 
	# 				self.inverted_index[t] = dict()
	# 			doc_term_positions = {docid: list()} 
	# 			# -- find position of each token 
	# 			term_positions = list(index for index,x in enumerate(words) if t in stemmer.convert_to_stem(x).split("&"))
	# 			tp = sorted(list(set(doc_term_positions[docid]+term_positions)))
	# 			doc_term_positions[docid] = ",".join(tp)
				
	# 			# -- add doc_term_positions to docs
	# 			self.inverted_index[t].update(doc_term_positions)
		
		
	
	# 	print("--- {} seconds ---".format(time.time() - start_time))

	# 	# write block to temp inverted indext file
	# 	# with open('inverted_index500.txt', 'w') as f:
	# 	# 	f.write(str(self.inverted_index))
	# 	# f.close()


	def open_inverted_index(self):
		start_time = time.time()
		inv_file = open("inverted_index2000.txt", "r")
		inv_string = inv_file.read()
		self.inverted_index = ast.literal_eval(inv_string)
		inv_file.close()

		print(len(self.inverted_index))

		print("--- {} seconds ---".format(time.time() - start_time))
		

	def get_query(self):
		os.system('clear')
		print('\033[94m'+ "=======================================" + '\033[0m')
		query = input(" 🔎 : ")
		print('\033[94m'+ "=======================================" + '\033[0m')
		# normalizer = Normalizer(statistical_space_correction=True)
		# tokenizer = Tokenizer()
		# stemmer = FindStems()
		# myspell_checker = SpellCheck()

		if query == "EXIT":
			return False
		
		block_list, block_tokens, block_words = list(), list(), list()
		must_include, must_include_tokens, must_include_words = list(), list(), list()
		

		if '!' in query:
			block = query.split("!")[1]
			query = query.split("!")[0]

			block_tokens, block_words = block.split(" "),  block.split(" ")
			# block_tokens, block_words = self.tokenizer(block)
			for bt in block_tokens:
				if bt in self.inverted_index:
					block_list = block_list + list(self.inverted_index[bt].keys())

		result = re.search('"(.*)"', query)
		if result == None:
			pass
		else:
			result = result.group(1)			
			must_include_tokens, must_include_words = result.split(" "),  result.split(" ")
			# must_include_tokens, must_include_words = self.tokenizer(result)
			must_include = tuple(must_include_tokens)
			


		# query = myspell_checker.spell_corrector(query)
		query_tokens, query_words = query.split(" "),  query.split(" ")
		# query_tokens, query_words = self.tokenizer(query)
		query_inverted_index = dict()

		for qt in query_tokens:
			if qt in self.inverted_index:
				query_inverted_index[qt] = self.inverted_index[qt]		
		# print(query_inverted_index)
		query_set = list(query_inverted_index.keys())		
		# print(query_set)

		
		# create combinatinos of query set
		combinations = list()
		for i in range(1,len(query_set)+1):
			for comb in itertools.combinations(query_set, i):
				combinations.append(comb)
		combinations = combinations[::-1]


		# find intersections of querys
		query_combination_priorty = dict()
		for comb in combinations:
			intersection_list = list(query_inverted_index[comb[0]].keys())
			for i in range(1,len(comb)):
				next_docs = list(query_inverted_index[comb[i]].keys())
				intersection_list = self.intersection(intersection_list, next_docs)

			forbidden = self.intersection(block_list, intersection_list)
			for fr in forbidden:
				intersection_list.remove(fr)			

			must_include_check = True
			for mi in must_include:
				if mi not in comb:
					must_include_check = False
					break

			if must_include_check:	
				# give priority to docs 
				if len(comb)>1:
					intersection_list = self.doc_priorty(must_include, intersection_list, must_include, query_inverted_index)
				query_combination_priorty[comb] = intersection_list
			else:
				# give priority to docs 
				if len(comb)>1:
					intersection_list = self.doc_priorty(comb, intersection_list, must_include, query_inverted_index)
				query_combination_priorty[comb] = intersection_list
		# print(query_combination_priorty)
		

		# final docs arrangment 
		response = list()
		for qc in query_combination_priorty:
			response = response + query_combination_priorty[qc]
		response = list(dict.fromkeys(response))
		
		# show 10 record of docs
		if response == []:
			print(" ☹️  no match result!")
		else:
			for doc_res in response[0:5]:
				print("🔻 [{}] [{}] [{}]".format(doc_res, self.db[doc_res]['title'],  self.db[doc_res]['url']))
				print("---------------------------------------")
		print('\033[94m'+ "=======================================" + '\033[0m')
		
		input("\t\t  +++ Press [Enter] +++")
		return True	

		
	def intersection(self, lst1, lst2):
		return list(set(lst1) & set(lst2))


	def doc_priorty(self, comb, idocs, mst, qii):
		docs_priorty = dict(zip(idocs, [0 for i in range(len(idocs))] ))
		
		for d in idocs:
			priorty = 0			
			for i in range(len(comb)-1):
				curr_q = comb[i]
				next_q = comb[i+1]
					
				# -- binaty check for combinations

				curr_q_positino = qii[curr_q][d].split(',')
				next_q_position = qii[next_q][d].split(',')

				min_dis = None
				for cpos in curr_q_positino:
					for npos in next_q_position:
						dis = abs(int(cpos) - int(npos))
					if min_dis == None:
						min_dis = dis
					if min_dis > dis:
						min_dis = dis
				priorty = priorty + min_dis			
			docs_priorty[d] = priorty

		# sort base on priorty, min value is first priorty
		docs_priorty = dict(sorted(docs_priorty.items(), key=lambda item: item[1]))
		# print(docs_priorty)
		return list(docs_priorty.keys())

	def heaps_law(self):
		
		
		x = []
		y = []
		token = 0
		for i in range(1, len(freq)):
			x.append(math.log10(i))
			token += freq[i - 1]
			y.append(math.log10(token))

		area = np.pi * 3
		# Plot
		plt.scatter(x, y, s=area, alpha=0.5)

		plt.title('Comaparison dataset columns')
		plt.xlabel('vocab size')
		plt.ylabel('number of tokens')

		plt.plot()
		plt.show()


	def zipf_law(self):
		freq = list()
		for w in self.inverted_index:
			freq.append(len(self.inverted_index[w].keys()))
		x = []
		y = []
		for i in range(len(freq)):
			x.append(math.log10(freq[i]))
			y.append(math.log10(freq[0] / freq[i]))

		area = np.pi * 3
		plt.scatter(x, y, s=area, alpha=0.5)

		plt.title('Comaparison dataset columns')
		plt.xlabel('log 10 rank')
		plt.ylabel('log 10 cfi')

		plt.plot()
		plt.show()


	def tf(t,d):
		pass

	def idf(t,D):
		pass

	def tfidf(t,d,D):
		pass


if __name__ == "__main__":
	json_path = [
				 'data0.json',
				 'data1.json',
				 'data2.json',
				 'data3.json',
				 'data4.json',
				 'data5.json',
				 'data6.json',
				 'data7.json',
				 'data8.json',
				 'data9.json',
				 'data10.json',
				 'data11.json',
				 'data12.json',
				]


	br = Browser(json_path)
	br.open_inverted_index()

	# br.heaps_law()
	# br.zipf_law()
	# print(len(br.inverted_index))

	q = True
	while(q !=False):
		q = br.get_query()

	