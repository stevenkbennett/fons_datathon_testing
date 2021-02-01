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
