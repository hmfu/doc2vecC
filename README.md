# doc2vecC
doc2vecC model for document classification

Main difference from the paper:

	1.	Count of words are used here for BOW. The author used boolean values, 1 for exist and 0 for not.
	
	2.	Fixed number of words are sampled from a document here. The author sampled by a fixed proportion. Specifically, 0.1.