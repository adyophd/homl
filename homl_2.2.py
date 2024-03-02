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
    # returns an array of booleans (resulting from logical evaluation of crc32 checksum against the test_ratio)

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio)) # lambda is a sort of for loop
    # ^ applies is_id function to the id_ data resulting in series of yes/nos to indicate membership to test set
    return data.loc[~in_test_set], data.loc[in_test_set]  # ~ means not

# the above functions ensure that identified data maps to the correct train/test set
# but the functions need an identifier

housing_with_id = housing.reset_index() # adds an index column
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
# Using an index column has downsides, such as ensuring new data always gets appended and no entries are deleted

# A better method is to use data that won't change over time, like lat/long
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]  # why *1000? ensure unique ids (no two 0s)
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")



