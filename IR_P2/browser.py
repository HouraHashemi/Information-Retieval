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
import itertools


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
	docs_tfidf_dict = dict()

	def __init__(self, json_path):
		# pass

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

		# small_inverted_index = dict(itertools.islice(self.inverted_index.items(), 500))
		# with open('inverted_index500.txt', 'w') as f:
		# 	f.write(str(small_inverted_index))
		# f.close()

		print(len(self.inverted_index))

		print("--- {} seconds ---".format(time.time() - start_time))
		

	def docs_tfidf(self):
		for t in self.inverted_index:
			docs = self.inverted_index[t].keys()
			nt = len(docs)
			D = len(self.db.keys())
			
			for d in docs:
				if d not in self.docs_tfidf_dict:
					self.docs_tfidf_dict[d] = list()
				tf = 1 + math.log(len(self.inverted_index[t][d]))
				idf = math.log((D/nt))
				tf_idf =  tf * idf
				self.docs_tfidf_dict[d].append([t,tf_idf])
		self.docs_tfidf_normalize()



	def docs_tfidf_normalize(self):
		alpha = 0
		for doc in self.docs_tfidf_dict:
			for term in self.docs_tfidf_dict[doc]:
				token = term[0]
				token_tfidf = term[1]

				alpha = alpha + token_tfidf ** 2
			alpha = alpha ** 0.5

		for doc in self.docs_tfidf_dict:
			for i in range(len(self.docs_tfidf_dict[doc])):
				self.docs_tfidf_dict[doc][i][1] = self.docs_tfidf_dict[doc][i][1]/alpha
				self.docs_tfidf_dict[doc][i] = tuple(self.docs_tfidf_dict[doc][i])
			self.docs_tfidf_dict[doc] = dict(self.docs_tfidf_dict[doc])



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
			

		# query = myspell_checker.spell_corrector(query)
		query_tokens, query_words = query.split(" "), query.split(" ")
		# query_tokens, query_words = self.tokenizer(query)
		query_inverted_index = dict()
		
		# for qt in query_tokens:
		# 	if qt in self.inverted_index:
		# 		query_inverted_index[qt] = self.inverted_index[qt]		
		# print(query_inverted_index)

		query_set = query_tokens
		print(query_set)


		# create tfidf
		query_tfidf = list()
		for t in query_set:
			freq = query_set.count(t)
			nt = len(self.inverted_index[t].keys())
			D = len(query_set)
			tf = 1 + math.log(freq)
			idf = math.log((D/nt))
			tf_idf = tf * idf
			query_tfidf.append([t,tf_idf])


		# normalizing query tfidf
		beta = 0
		for q in range(len(query_tfidf)):
			beta = beta + query_tfidf[q][1]**2
		beta = beta ** 0.5
		for q in range(len(query_tfidf)):
			query_tfidf[q][1] = query_tfidf[q][1]/beta
			query_tfidf[q] = tuple(query_tfidf[q])
		query_tfidf = dict(list(set(query_tfidf)))



		extracted_docs = dict()
		for query in query_tfidf:
			if query in self.inverted_index:
				docs = self.inverted_index[query].keys()
				for doc in docs:
					if doc not in extracted_docs:
						extracted_docs[doc] = 0
					extracted_docs[doc] = extracted_docs[doc] + (self.docs_tfidf_dict[doc][query] * query_tfidf[query])

		extracted_docs = {k: v for k, v in sorted(extracted_docs.items(), key=lambda item: item[1])}
		list(extracted_docs.items()).reverse()
		response = list(dict(extracted_docs).keys())

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


	def doc_priorty(self, comb, idocs):
		docs_priorty = dict(zip(idocs, [0 for i in range(len(idocs))] ))

		for token in comb:
			for doc in idocs:
				tf = self.tf(token,doc)
				weight = self.tf_idf(token,doc)


	def doc_priorty_p1(self, comb, idocs, mst, qii):
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
	br.docs_tfidf()
	
	# br.heaps_law()
	# br.zipf_law()
	# print(len(br.inverted_index))

	q = True
	while(q !=False):
		q = br.get_query()

	