
import pandas as pd
import pickle, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed


from matminer.featurizers.base import MultipleFeaturizer

# compositional featurizers
from matminer.featurizers.composition import (
    ## composite
    ElementProperty,
    ## element
    Stoichiometry,
    ## ion
    IonProperty, ElectronAffinity,
    ## orbital
    AtomicOrbitals, ValenceOrbital
)

#structural featurizers
from matminer.featurizers.structure import (
    # bonding
    StructuralHeterogeneity,
    # order
    DensityFeatures,
)


# site featurizers
# from matminer.featurizers.site.chemical import (
#     #chemical
#     ChemicalSRO
# )



# compositional featurizers
element_featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
stoich_featurizer = Stoichiometry()
ion_prop_featurizer = IonProperty(impute_nan=True)
# e_affinity_featurizer = ElectronAffinity()
atomic_orb_featurizer = AtomicOrbitals()
valence_orb_featurizer = ValenceOrbital(impute_nan=True)
# structural
struct_het_featurizer = StructuralHeterogeneity()
density_featurizer = DensityFeatures()
# chemical_featurizer = ChemicalSRO(nn=6).fit(data["structure"])


structural_featurizer = MultipleFeaturizer([
    density_featurizer,
    # struct_het_featurizer,
])

compositional_featurizer = MultipleFeaturizer([
    stoich_featurizer,
    ion_prop_featurizer,
    # e_affinity_featurizer,
    # atomic_orb_featurizer,
    valence_orb_featurizer,
    element_featurizer,
])

def featurize_chunk(chunk):
    # composition = pickle.loads(composition_bytes)

    """
    Featurize a chunk of data.

    Parameters
    ----------
    chunk : pd.DataFrame
        A chunk of the data to be featurized. Must contain columns 'structure' and 'composition'.

    Returns
    -------
    list
        A list of length 2, where the first element is a pd.DataFrame containing structural features,
        and the second element is a pd.DataFrame containing compositional features.
    """
    structural_features = structural_featurizer.featurize_dataframe(chunk, col_id="structure", ignore_errors=True, inplace=False)
    compositional_features = compositional_featurizer.featurize_dataframe(chunk, col_id="composition", ignore_errors=True, inplace=False)
    
    # Combine features
    return [structural_features, compositional_features]



def main():
    n_workers =int(sys.argv[1])
    save_after_n_chunks = n_workers
    
    data_path = "matbench_mp_gap_raw.data"
    featurized_path = "matbench_mp_gap_featurized.data"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

        
    data['composition'] = data['structure'].apply(lambda struct: struct.composition)
    
    chunk_size = int(sys.argv[2])
    print(f"Chunk size: {chunk_size}")
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    folder_path = "features"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    num_groups = (len(chunks) + n_workers - 1) // n_workers  # Ensure all groups are created
    groups = {i: {"struct": [], "comp": []} for i in range(num_groups)}


    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        
        # Submit all chunks immediately
        for i, chunk in enumerate(chunks):
            future = executor.submit(featurize_chunk, chunk)
            futures[future] = i
        

            
        for future in as_completed(futures):
            chunk_index = futures[future]
            try:
                structural_features, compositional_features = future.result()
                print(f"Finished processing chunk {chunk_index + 1}/{len(chunks)}", flush=True)

                # Collect results
                if len(groups[chunk_index // n_workers]["struct"]) == 0:
                    groups[chunk_index // n_workers]["struct"] = [None] * n_workers
                    groups[chunk_index // n_workers]["comp"] = [None] * n_workers
                    
                groups[chunk_index // n_workers]["struct"][chunk_index % n_workers] = structural_features
                groups[chunk_index // n_workers]["comp"][chunk_index % n_workers] = compositional_features

                # Check if we have reached the save threshold
                for i, val in groups.items():
                    if len(val["struct"]) == n_workers and all(isinstance(s, pd.DataFrame) for s in val["struct"]):

                        # Combine features into a DataFrame
                        df_structural_features = pd.concat(val["struct"], axis=0)
                        df_compositional_features = pd.concat(val["comp"], axis=0)
                        df_featurized = pd.concat([data.iloc[ i * (chunk_size) * n_workers:(i+1)*(chunk_size)*(n_workers)], 
                                                    df_structural_features, 
                                                    df_compositional_features], axis=1)

                        # Save the features for this batch
                        batch_file_path = os.path.join(folder_path, f"{folder_path}_{i}.features")
                        with open(batch_file_path, "wb") as f:
                            pickle.dump(df_featurized, f)
                        print(f"Saved features for chunks {i * n_workers} to {(i + 1) * n_workers} to {batch_file_path}")

                        groups[i] = {"struct": [], "comp": []}

            

            except Exception as e:
                print(f"Error processing chunk {chunk_index}: {e}")

    # Handle any remaining features if they exist
    for i, val in groups.items():
        if len(val["struct"]) > 0:
            for j in range(n_workers):
                if isinstance(val["struct"][j], type(None)):
                    break
            struct = val["struct"][:j]
            comp = val["comp"][:j]
            df_structural_features = pd.concat(struct, axis=0)
            df_compositional_features = pd.concat(comp, axis=0)
            df_featurized = pd.concat([data.iloc[ i* (chunk_size) * n_workers: (i + 1)*(chunk_size) * (n_workers)], 
                                        df_structural_features,  
                                        df_compositional_features], axis=1)

            # Save the features for this batch
            batch_file_path = os.path.join(folder_path, f"{folder_path}_{i}.features")
            with open(batch_file_path, "wb") as f:
                pickle.dump(df_featurized, f)
            print(f"Saved features for chunks {i * n_workers} to {len(chunks) - 1} to {batch_file_path}")


  
    # with open(featurized_path, "wb") as f:
    #     pickle.dump(df_featurized, f)
        
        
if __name__ == "__main__":
    main()

    # data_path = "matbench_mp_gap_raw.data"
    # featurized_path = "matbench_mp_gap_featurized.data"
    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)

    # structural_features = structural_featurizer.featurize_dataframe(data, col_id="structure", ignore_errors=True, inplace=False)
    # compositional_features = compositional_featurizer.featurize_dataframe(data, col_id="composition", ignore_errors=True, inplace=False)
    
    # # Combine features
    # df_featurized = pd.concat([data, structural_features, compositional_features], axis=1)
    # with open(featurized_path, "wb") as f:
    #     pickle.dump(df_featurized, f)