
import pandas as pd
import pickle, os, sys
from concurrent.futures import ProcessPoolExecutor


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

def featurize_chunk(chunk, i):
    # composition = pickle.loads(composition_bytes)

    structural_features = structural_featurizer.featurize_dataframe(chunk, col_id="structure", ignore_errors=True, inplace=False)
    compositional_features = compositional_featurizer.featurize_dataframe(chunk, col_id="composition", ignore_errors=True, inplace=False)
    
    # Combine features
    return [structural_features, compositional_features]



def main():
    n_workers =int(sys.argv[1])
    
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

  
    with ProcessPoolExecutor(max_workers= n_workers) as executor:
        features = list(executor.map(featurize_chunk, chunks, range(n_workers)))
    

    structural_features_list, compositional_features_list = zip(*features)
    df_structural_features = pd.concat(structural_features_list, axis=0)
    df_compositional_features = pd.concat(compositional_features_list, axis=0)

    # Combine features
    df_featurized = pd.concat([data, df_structural_features, df_compositional_features], axis=1)

    with open(featurized_path, "wb") as f:
        pickle.dump(df_featurized, f)

    

  
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