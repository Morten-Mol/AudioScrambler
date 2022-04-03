"""Script for comparing the index boundaries of the bands to be shuffled. They should be equal for the
scrambling and descrambling operation.

Getting the files needed to run this debug script is done by uncommenting the last lines
of the calc_band_boundaries() function in the scrambler.py module and running pytest on the project.
Also make sure that the Recording.wav is the only sound file present in the project when creating
the data to be debugged.

Running the test once with the debugging enabled will create a index_numbers file in the debugging folder, which
can be renamed to index_numbers_scram.txt or index_numbers_descram.txt, depending on in which step of the process
the file was created.

After doing this the below script will check that all frequency band index boudaries are similar in both scrambling
and descrambling.
"""
import numpy as np

# Save scram numbers from file
from operator import index


index_scram = []

with open('index_numbers_scram.txt', 'r') as file_scram:
    for line in file_scram:
        lin = line.split(',')
        index_scram.append([int(lin[0]), int(lin[1][:-1])])

index_descram = []

with open('index_numbers_descram.txt', 'r') as file_descram:
    for line in file_descram:
        lin = line.split(',')
        index_descram.append([int(lin[0]), int(lin[1][:-1])])

# Flip the index_descram list

index_descram = index_descram[::-1]

# Flip order for each set of edge index
for i in np.arange(0, len(index_descram), 2):
    i = int(i)
    first = index_descram[i]
    second = index_descram[i+1]
    index_descram[i] = second
    index_descram[i+1] = first


# compare index numbers
for i_scram, i_descram in zip(index_scram, index_descram):
    assert i_scram[0] == i_descram[0]
    assert i_scram[1] == i_descram[1]
