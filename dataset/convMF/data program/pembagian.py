import sys
from data_manager import Data_Factory

do_preprocess = True
data_path = 'data/data'
aux_path = 'data/aux'
data_factory = Data_Factory()

path_rating = 'data/data_rating.txt'
path_itemtext = 'data/data_document.txt'
min_rating = 30
max_length = 300
max_df = 0.5
vocab_size = 8000
split_ratio = 0.2

print ("=================================Preprocess Option Setting=================================")
print ("\tsaving preprocessed aux path - %s" % aux_path)
print ("\tsaving preprocessed data path - %s" % data_path)
print ("\trating data path - %s" % path_rating)
print ("\tdocument data path - %s" % path_itemtext)
print ("\tmin_rating: %d\n\tmax_length_document: %d\n\tmax_df: %.1f\n\tvocab_size: %d\n\tsplit_ratio: %.1f" % (min_rating, max_length, max_df, vocab_size, split_ratio))
print ("===========================================================================================")

R, D_all = data_factory.preprocess(path_rating, path_itemtext, min_rating, max_length, max_df, vocab_size)
data_factory.save(aux_path, R, D_all)
data_factory.generate_train_valid_test_file_from_R(data_path, R, split_ratio)
