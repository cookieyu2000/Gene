import h5py

# Open the input text file
with open('data/PubMed_ner_data.txt', 'r') as file:
    data = file.read()

# Convert the data to a numpy array or list (depending on your data structure)
# For example, assuming each line in the text file represents a data point:
data_list = data.split('\n')

# Create a new h5 file
with h5py.File('data/PubMed_ner_data.h5', 'w') as h5file:
    # Create a dataset in the h5 file
    dataset = h5file.create_dataset('data', data=data_list)

# Print a success message
print("Data successfully converted to h5 format.")