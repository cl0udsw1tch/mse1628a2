
import pandas as pd
import pickle, os
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


compositional_featurizer = MultipleFeaturizer([
    stoich_featurizer,
    ion_prop_featurizer,
    # e_affinity_featurizer,
    # atomic_orb_featurizer,
    valence_orb_featurizer,
    # element_featurizer,
])
structural_featurizer = MultipleFeaturizer([
    #density_featurizer,
    struct_het_featurizer,
])

def featurize_chunk(chunk, i):
    # composition = pickle.loads(composition_bytes)

    structural_featurizer.featurize_dataframe(chunk, col_id="structure", ignore_errors=True, inplace=True)
    compositional_featurizer.featurize_dataframe(chunk, col_id="composition", ignore_errors=True, inplace=True)
    
    # Combine features
    print(f"Processed chunk: {i}")
    return chunk



def main():
    data_path = "matbench_mp_gap_raw.data"
    featurized_path = "matbench_mp_gap_featurized2.data"
    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)
    with open("matbench_mp_gap_featurized.data", "rb") as f:
        data = pickle.load(f)
        
    # data['composition'] = data['structure'].apply(lambda struct: struct.composition)
    # compositional_bytes = data['composition'].apply(lambda x: pickle.dumps(x))
    
    non_magpie = data.copy().iloc[:, :7]

    
    chunk_size = 128
    chunks = [non_magpie[i:i + chunk_size] for i in range(0, 2*chunk_size, chunk_size)]

    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as executor:
        features = pd.concat(list(executor.map(featurize_chunk, chunks, [0,1])), axis=0)
        
    df_features = pd.DataFrame(features, 
                               columns= structural_featurizer.feature_labels() + 
                                        compositional_featurizer.feature_labels())
    df_featurized = pd.concat([non_magpie.iloc[:2 * chunk_size, :7], df_features, data.iloc[:2*chunk_size, 7:]], axis=1)

    with open(featurized_path, "wb") as f:
        pickle.dump(df_featurized, f)
        
        
if __name__ == "__main__":
    main()