
import numpy as np
import re
import random
import os
import pickle as pkl


class Imdb_loader(object):


	def __init__(self):
		
		self.train_word_list_list = []
		self.train_label_list_list = []
		self.val_word_list_list = []
		self.val_label_list_list = []
		self.unlabeled_word_list_list = []
		self.unlabeled_label_list_list = []




	def load_text(self, folder_name, file_tail, label, roll, max_samps):

		word_list_list = []
		label_list_list = []


		for file_name in os.listdir(folder_name):

			if not file_name.endswith(file_tail):
				continue


			with open(os.path.join(folder_name, file_name), 'r') as f:

				line_count = 0

				for line in f:
					string = line.rstrip().lower()
					word_list_list += [[word for word in re.findall(r"[\w]+|[.,!?;']", string) if len(word) != 0]]
					label_list_list += [[label]]

					line_count += 1

				if len(word_list_list) == max_samps:
					break

				if line_count != 1:
					raise Exception('file %s has line count not equal to 1.' % os.path.join(folder_name, file_name))

		if roll == 'train':
			self.train_word_list_list += word_list_list
			self.train_label_list_list += label_list_list
		elif roll == 'val':
			self.val_word_list_list += word_list_list
			self.val_label_list_list += label_list_list
		elif roll == 'unlabeled':
			self.unlabeled_word_list_list += word_list_list
			self.unlabeled_label_list_list += label_list_list

		

		print ('train samps, labels:	', len(self.train_word_list_list), len(self.train_label_list_list))
		print ('val samps, labels:	', len(self.val_word_list_list), len(self.val_label_list_list))
		print ('unlabeled samps, labels:	', len(self.unlabeled_word_list_list), len(self.unlabeled_label_list_list))

		# print ('word_list_list:	', word_list_list[0])





	def build_vocab2idx(self, count_thres):

		word_list_list = self.train_word_list_list + self.val_word_list_list + self.unlabeled_word_list_list

		vocab2count = {}

		for word_list in word_list_list:

			# if len(word_list) > 8000:
			# 	print (len(word_list), word_list)

			for word in word_list:
				vocab2count[word] = vocab2count.get(word, 0) + 1

		raw_vocabs = len(vocab2count)



		vocab2idx = {}
		next_idx = 2 # 0 for rare words, 1 for padding.

		for vocab in vocab2count.keys():

			if vocab2count[vocab] < count_thres:
				continue

			vocab2idx[vocab] = next_idx
			next_idx += 1

		self.vocab2idx = vocab2idx
		self.vocabs = len(vocab2idx) + 2

		print ('raw_vocabs, vocabs:	', raw_vocabs, self.vocabs)





	def operate_vocab2idx(self, include_unseen):

		if include_unseen:

			self.train_word_list_list = [[self.vocab2idx.get(word, 0) for word in word_list] \
					for word_list in self.train_word_list_list]
			self.val_word_list_list = [[self.vocab2idx.get(word, 0) for word in word_list] \
					for word_list in self.val_word_list_list]
			self.unlabeled_word_list_list = [[self.vocab2idx.get(word, 0) for word in word_list] \
					for word_list in self.unlabeled_word_list_list]

		else:

			vocab_set = set(self.vocab2idx.keys())

			self.train_word_list_list = [[self.vocab2idx[word] for word in word_list if word in vocab_set] \
					for word_list in self.train_word_list_list]
			self.val_word_list_list = [[self.vocab2idx[word] for word in word_list if word in vocab_set] \
					for word_list in self.val_word_list_list]
			self.unlabeled_word_list_list = [[self.vocab2idx[word] for word in word_list if word in vocab_set] \
					for word_list in self.unlabeled_word_list_list]






	def build_rep_data(self, context_len, doc_samp_len, include_val, val_pro, target_at_middle):

		if target_at_middle and context_len % 2 != 0:
			raise Exception('Context_len should be even if target_at_middle is used.')

		self.context_len = context_len
		self.doc_samp_len = doc_samp_len


		word_list_list = self.train_word_list_list + self.unlabeled_word_list_list

		if include_val:
			word_list_list += self.val_word_list_list


		window_size = context_len + 1

		context_list_list = []
		masked_list_list = []
		doc_samp_list_list = []

		samp_count = 0

		for word_list in word_list_list:

			if samp_count % 10000 == 0:
				print ('building rep data for sample ', samp_count)
			samp_count += 1

			length = len(word_list)

			for start_idx in range(length - window_size):

				if target_at_middle:
					context_list = word_list[start_idx: start_idx + window_size]
					
					masked_list_list += [[context_list.pop(window_size // 2)]]
					context_list_list += [context_list]
					doc_samp_list_list += [random.sample(word_list, doc_samp_len)]

				else:

					for masked_idx in range(window_size):

						context_list = word_list[start_idx: start_idx + window_size]
						
						masked_list_list += [[context_list.pop(masked_idx)]]
						context_list_list += [context_list]

						if doc_samp_len <= len(word_list):
							doc_samp_list_list += [random.sample(word_list, doc_samp_len)]
						else:
							doc_samp_list_list += [word_list * (doc_samp_len // len(word_list)) \
									+ random.sample(word_list, doc_samp_len % len(word_list))]

		print ('shuffling the samples...')
		masked_list_list, context_list_list, doc_samp_list_list \
				= self.shuffle_list_list([masked_list_list, context_list_list, doc_samp_list_list])

		context_arr = np.array(context_list_list)
		masked_arr = np.array(masked_list_list)
		doc_samp_arr = np.array(doc_samp_list_list)

		total_samps = len(context_arr)
		val_samps = int(total_samps * val_pro)

		self.val_context_arr = context_arr[:val_samps]
		self.val_masked_arr = masked_arr[:val_samps]
		self.val_doc_samp_arr = doc_samp_arr[:val_samps]
		self.train_context_arr = context_arr[val_samps:]
		self.train_masked_arr = masked_arr[val_samps:]
		self.train_doc_samp_arr = doc_samp_arr[val_samps:]



		print ('number of rep samples:	', len(context_list_list))
		print ('val samps, train samps:	', len(self.val_context_arr), len(self.train_context_arr))



	def build_tuning_data_arr(self):

		self.train_word_arr = np.array(self.train_word_list_list)
		self.train_label_arr = np.array(self.train_label_list_list)
		self.val_word_arr = np.array(self.val_word_list_list)
		self.val_label_arr = np.array(self.val_label_list_list)

		# train_lab2count = {}
		# for label in self.train_label_arr:
		# 	train_lab2count[label[0]] = train_lab2count.get(label[0], 0) + 1
		# val_lab2count = {}
		# for label in self.val_label_arr:
		# 	val_lab2count[label[0]] = val_lab2count.get(label[0], 0) + 1

		# print ('	train_lab2count:	', train_lab2count)
		# print ('	val_lab2count:	', val_lab2count)




	def save_object_attribute_arr(self, file_name):

		with open(file_name, 'wb') as f:

			print ('saving object attribute...')
			
			pkl.dump((self.context_len, self.doc_samp_len, self.vocabs, \
					self.train_context_arr, self.train_doc_samp_arr, self.train_masked_arr, \
					self.val_context_arr, self.val_doc_samp_arr, self.val_masked_arr, \
					self.train_word_arr, self.train_label_arr, \
					self.val_word_arr, self.val_label_arr), f, protocol = 4)

			print ('object attribute arr saved as ', file_name)




	def load_object_attribute(self, file_name):

		with open(file_name, 'rb') as f:

			print ('loading object attribute...')

			self.context_len, self.doc_samp_len, self.vocabs, \
					self.train_context_arr, self.train_doc_samp_arr, self.train_masked_arr, \
					self.val_context_arr, self.val_doc_samp_arr, self.val_masked_arr, \
					self.train_word_arr, self.train_label_arr, \
					self.val_word_arr, self.val_label_arr = pkl.load(f)

			print ('object attribute loaded from ', file_name)
			







	def shuffle_list_list(self, list_list):

		if len(set([len(list_) for list_ in list_list])) != 1:
			raise Exception('list of different lenths found in list_list.')

		order = np.random.permutation(len(list_list[0]))

		return tuple([np.array(list_)[order].tolist() for list_ in list_list])




	def link_list_list(self, list_list):

		output_list = []

		for list_ in list_list:
			output_list += list_

		return output_list



