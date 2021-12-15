import os
import pandas as pd
import pickle
import pprint
import json

os.getcwd()
cur_dir = '/home/ncc/PycharmProjects/PETCT'

# pickle_dir = os.path.join(cur_dir, 'media/ncc/nnUNet_preprocessed/Task577_KidneyTumour')
pickle_dir = os.path.join(cur_dir, 'media/ncc/nnUNet_preprocessed/Task577_KidneyTumour/')
pickle_name = 'nnUNetPlansv2.1_plans_3D.pkl'
pkl_file = os.path.join(pickle_dir, pickle_name)
'''
json_dir = os.path.join(cur_dir, 'media/ncc/nnUNet_preprocessed/Task577_KidneyTumour')
json_name = 'splits_final.json'
json_file = os.path.join(json_dir, json_name)
'''


with open(pkl_file, 'rb') as f:
    props = pickle.load(f)

print(type(props))
print(props.keys()) # keys, values
print(props)

##

# Load pickle file
input_file = open(pkl_file, 'rb')
new_dict = pickle.load(input_file)
input_file()

# Create a Pandas DataFrame
data_frame = pd.DataFrame(new_dict)

# Copy DataFrame index as a column
data_frame['index'] = data_frame.index

# Move the new index column to the from of the DataFrame
index = data_frame['index']
data_frame.drop(labels=['index'], axis=1, inplace = True)
data_frame.insert(0, 'index', index)

# Convert to json values
json_data_frame = data_frame.to_json(orient='values', date_format='iso', date_unit='s')
with open('data.json', 'w') as js_file:
    js_file.write(json_data_frame)