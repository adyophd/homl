# --------------------------
# Create a test set
# --------------------------

import numpy as np

# IMPORTANT NOTE: This test set partition method is not optimal as it can lead to replication issues. See below.
def shuffle_and_split(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]  # data.iloc selects subset of data by its integer location

train_set, test_set = shuffle_and_split(housing,0.2)  # housing file from homl_2.1py
len(train_set) #16512
len(test_set) #4128

# Better implementation:

from zlib import crc32 # creates pseudo-random but consistent hashes of the identifiers for consistent data splitting

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32  # 2**3 references the max checksum value (32 bit)
    # getting the checksum values that fall within the range specified by test_ratio

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    # ^ applies is_id function to the id_ data resulting in series of yes/nos to indicate membership to test set
    return data.loc[~in_test_set], data.loc[in_test_set]  # ~ means not




