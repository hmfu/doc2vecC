
import sys, getopt
sys.path.insert(0, './sources')

from doc2vecC_model import Doc2vecC_model
from enwiki100_loader import Enwiki100_loader
import numpy as np



if __name__ == '__main__':

	enwiki100_loader = Enwiki100_loader()

	enwiki100_loader.load_text('./datasets/enwiki100/alldata.txt', max_samps = None)
	enwiki100_loader.load_label('./datasets/enwiki100/alldata-label.txt', max_samps = None)
	
	enwiki100_loader.build_vocab2idx(count_thres = 10)
	enwiki100_loader.operate_vocab2idx()

	enwiki100_loader.split_data(train_samps_per_lab = 100, val_samps_per_lab = 900)

	enwiki100_loader.build_rep_data(context_len = 6, doc_samp_len = 12, include_val = False, val_pro = 0.05, target_at_middle = False)
	enwiki100_loader.build_tuning_data_arr()


	data_loader = enwiki100_loader





	doc2vecC_model = Doc2vecC_model(weight_stddev = 0.024, \
									bias_stddev = 0.00001, \
									gpu_idx = 1, \
									float_type = 'float32')


	doc2vecC_model.build_model(context_len = data_loader.context_len, \
								doc_samp_len = data_loader.doc_samp_len, \
								vocabs = data_loader.vocabs, \
								embed_dims = 100, \
								neg_samps = 5)


	doc2vecC_model.train_model(learning_rate = 0.001, \
								epochs = 10000, \
								batch_size = 1024, \

								train_context_arr = data_loader.train_context_arr, \
								train_masked_arr = data_loader.train_masked_arr, \
								train_doc_samp_arr = data_loader.train_doc_samp_arr, \
								val_context_arr = data_loader.val_context_arr, \
								val_masked_arr = data_loader.val_masked_arr, \
								val_doc_samp_arr = data_loader.val_doc_samp_arr, \

								train_word_arr = data_loader.train_word_arr, \
								train_label_arr = data_loader.train_label_arr, \
								val_word_arr = data_loader.val_word_arr, \
								val_label_arr = data_loader.val_label_arr, \

								dropout_keep_prob = 0.1, \
								epoch_per_eval = 5, \
								batch_per_print = 1000, \
								model_save_path = './models/model.ckpt', \
								print_samps = 3)




