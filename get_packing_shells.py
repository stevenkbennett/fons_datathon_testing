import pickle
from ccdc.io import CrystalReader, CrystalWriter

# Create CrystalReader Instance
csd_reader = CrystalReader()

# Load Data and get idxs
with open('search_results.pickle','rb') as pickle_in:
    data = pickle.load(pickle_in)
idxs = data['identifier']

# Get packed_shells
packed_shells = []
for idx in idxs:
    crystal = csd_reader.crystal(idx)
    packed = crystal.packing_shell(packing_shell_size=12)
    packed_shells.append(packed)

# Write to *.cif file
with CrystalWriter('packed_shells.cif') as crystal_writer:
    for shell in packed_shells:
        crystal_writer.write(shell)
