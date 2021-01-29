import ccdc.io as io
import pandas as pd
import pickle

def main():
    data_dict = {'identifier':[],'n_heavy_atoms':[],'smiles':[],
             'a':[],'b':[],'c':[],
             'alpha':[],'beta':[],'gamma':[],
             'z_value':[],'z_prime':[],'spacegroup_symbol':[],
             'contacts':[],'cell_volume':[], 'calculated_density':[],
             'packing_coefficient':[],'void_volume':[],'is_centrosymmetric':[],
             'is_sohncke':[]}
    mols = []
    count = 0
    csd_reader = io.EntryReader()
    for entry in csd_reader:
        crystal = entry.crystal
        mol = crystal.molecule
        is_single_molecule = len(mol.components) == 1
        only_15_heavy = len(mol.heaviest_component.heavy_atoms) <= 15
        is_3d = mol.heaviest_component.is_3d == True
        is_organic = mol.heaviest_component.is_organic == True
        not_polymeric = mol.heaviest_component.is_polymeric == False
        is_organometallic = mol.heaviest_component.is_organometallic == False
        no_disorder = crystal.has_disorder == False

        criteria_met = (is_single_molecule & only_15_heavy & is_3d & is_organic & not_polymeric & is_organometallic & no_disorder)
        if criteria_met:
            # Get Contacts!!!!
            contact_dict = {}
            for contact in crystal.contacts():
                if contact.type not in contact_dict.keys():
                    contact_dict[contact.type] = {'intermolecular':0,
                                                  'intramolecular':0}
                if contact.intermolecular:
                    contact_dict[contact.type]['intermolecular'] +=1
                else:
                    contact_dict[contact.type]['intramolecular'] +=1

            for hbond in crystal.hbonds():
                if hbond.type not in contact_dict.keys():
                    contact_dict[hbond.type] = {'intermolecular':0,
                                                  'intramolecular':0}
                if hbond.intermolecular:
                    contact_dict[hbond.type]['intermolecular'] +=1
                else:
                    contact_dict[hbond.type]['intramolecular'] +=1

            data_dict['identifier'].append(entry.identifier)
            data_dict['smiles'].append(mol.heaviest_component.smiles)
            data_dict['n_heavy_atoms'].append(len(mol.heaviest_component.heavy_atoms))
            data_dict['a'].append(crystal.cell_lengths[0])
            data_dict['b'].append(crystal.cell_lengths[1])
            data_dict['c'].append(crystal.cell_lengths[2])
            data_dict['alpha'].append(crystal.cell_angles[0])
            data_dict['beta'].append(crystal.cell_angles[1])
            data_dict['gamma'].append(crystal.cell_angles[2])
            data_dict['z_value'].append(crystal.z_value)
            data_dict['z_prime'].append(crystal.z_prime)
            data_dict['spacegroup_symbol'].append(crystal.spacegroup_symbol)
            data_dict['cell_volume'].append(crystal.cell_volume)
            data_dict['calculated_density'].append(crystal.calculated_density)
            data_dict['packing_coefficient'].append(crystal.packing_coefficient)
            data_dict['void_volume'].append(crystal.void_volume())
            data_dict['is_centrosymmetric'].append(crystal.is_centrosymmetric)
            data_dict['is_sohncke'].append(crystal.is_sohncke)
            data_dict['contacts'].append(contact_dict)
            mols.append(mol.heaviest_component)

        if count % 10000 == 0:
            with open(f'ckpt_search_results.pickle','wb') as pickle_out:
                pickle.dump(data_dict,pickle_out)
            with io.MoleculeWriter(f'ckpt_small_molecule_search.mol2') as mol_writer:
                for mol in mols:
                    mol_writer.write(mol)
            pd.DataFrame(data_dict).set_index('identifier').to_csv(f'ckpt_search_dict_test.csv',index=True)

        count += 1

    with open('search_results.pickle','wb') as pickle_out:
        pickle.dump(data_dict,pickle_out)
    with io.MoleculeWriter('small_molecule_search.mol2') as mol_writer:
        for mol in mols:
            mol_writer.write(mol)
    pd.DataFrame(data_dict).set_index('identifier').to_csv('search_dict_test.csv',index=True)


if __name__ == '__main__':
    main()
