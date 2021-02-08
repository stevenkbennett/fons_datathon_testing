# fons_datathon
Setting up data for FONs datathon on learning crystal properties from small molecules

# Search Results
The Cambridge Structural Database was searched for all molecules that satisfied the following criteria (see search_small_molecules.py for code):
 - Have only one molecular component
 - Has up to and including 15 non-hydrogen atoms
 - Has 3D coordinates
 - Is organic
 - Is NOT polymeric
 - Has no disorder

This search returned 29374 hits. Further cleaning will be required to remove entries without SMILEs strings and repeat entries. The following crystal structure information was recorded for all hits:
 - Unit cell lengths (a, b, c) (angstroms)
 - Unit cell Angles (alpha, beta, gamma) (degrees)
 - Z Value
 - Z prime Value
 - Space-group symbol
 - Number of Contacts and Hydrogen Bonds
 - Cell Volume (angstroms^3)
 - Calculate Density  (g/cm^3)
 - Packing Coefficient
 - Void Volume
 - Is Centrosymmetric
 - Is Sohncke

This data is stored as a both a csv and a pickle file (search_dict_test.csv and search_results.pickle, respectively). The csv file is to enable the user to view that data, but should not be used for further data analysis, owing to the fact that pandas DataFrames save python dictionary entries (in this case the contacts columns) as strings. The raw python dictionary object (from which the DataFrame was created) containing all of the data is saved in the pickle file. The 3D coordinates of all small molecule hits are saved in small_molecule_search.mol2.

Packing shells have been calculated with a packing shell size of 12. All of the corresponding cif files can be found in packed_shells.cif file and were generated using the get_packing_shells.py file. Given the large size of the generated cif file (>100MB), the file can be accessed via the team box folder.

# Descriptor Calculations
Following the search of the CSD, descriptors were calculated using the Python package [Mordred](https://mordred-descriptor.github.io).
2D and 3D descriptors were calculated for 29,095 molecules searched from the CSD. Descriptor calculations failed for 279 molecules due to the failure to idenitfy rings whilst using the RDKit function `SanitizeMol`.
In total, 1613 2D descriptors and 213 3D descriptors were calculated for each molecule.
These decriptors range from atom counts, to logS values. A full list of all descriptors can be found [here](https://mordred-descriptor.github.io/documentation/master/descriptors.html), in addition to an explanation of each one.

The data is stored a pickled Pandas Dataframe, which can be loaded in Python with the `read_pickle` function from the Pandas library. 
The data can be obtained using this [link](https://imperialcollegelondon.box.com/v/fons-datathon-descriptors) using the password 'datathon2021'.
In this Dataframe, the SMILES strings for each molecule can be found in the "SMILES" column, in addition to the RDKit Molecule object in the "RDKit_Molecule" column. 
Each molecule has a unique index number, from 0 to 29,075. 


