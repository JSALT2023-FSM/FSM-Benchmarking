"""
This script is used to benchmark the composition operation in k2.
"""

repetitions = 500

import os
import torch
torch.set_num_threads(int(os.environ["MKL_NUM_THREADS"]))
# print if gpu is available
print("GPU available: ", torch.cuda.is_available())



import k2
import numpy as np
from kaldifst.utils import k2_to_openfst
import kaldifst
from pathlib import Path
import pandas as pd
import timeit
from tqdm import tqdm
import fire


def float2bytesint(x):
    return np.frombuffer(bytes(np.array(x, dtype=np.float32)),dtype=np.int32)

def txtpath2k2fsa(filename):
    return k2.Fsa.from_openfst(open(filename).read(), num_aux_labels=1,)

def ofstnumarcs(fsa):
    return sum([fsa.num_arcs(s) for s in range(fsa.num_states)])

print("Loading data...")
def main(device, dbname, other_db=None):

    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("GPU not available, using CPU instead")
    
    if other_db is not None:
        df = pd.read_csv(other_db)
    else:
        df = pd.read_csv(f"data/{dbname}_composed.csv")

    results = []
    cores = os.environ["MKL_NUM_THREADS"]
    output = f"results/{dbname}_k2_{device}_{cores}_compbenchs.csv"
    if Path(output).exists():
        print("Loading previous results...")
        df_doned = pd.read_csv(output)
        results += df_doned.to_dict(orient="records")
    else:
        df_doned = None

    df_doned = None

    # for i in tqdm(range(len(df))):
    for i in tqdm(range(1000)):
            
            r = df.iloc[i]
            if df_doned is not None and r["fileC"] in df_doned["fileC"].values:
                print("Skipping ",r["fileC"])
                continue
            try:
                A = txtpath2k2fsa(Path(r["fileA"]).with_suffix(".txt"))
            except:
                print("Error reading ",r["fileA"])
                continue
            
            try:
                B = txtpath2k2fsa(Path(r["fileB"]).with_suffix(".txt"))
            except:
                print("Error reading ",r["fileB"])
                continue

            if int(r["narcs"])>500000:
                _repetitions = 2

                # print("Skipping with many arcs",r["fileC"])
                # continue

            # if i==0:
                # _repetitions = 2
            else:
                _repetitions = repetitions

            A = A.to(device)
            B = B.to(device)


            # print("Composing...")
            C = k2.compose(k2.arc_sort(A),k2.arc_sort(B),treat_epsilons_specially=False)
            try:
                arcs = C.to('cpu').as_dict()['arcs']
                if len(arcs)>0:
                    num_states= arcs.max(0).values[0:2].max().item()
                    num_arcs = len(arcs)
                else:
                    num_states = 0
                    num_arcs = 0
            except:
                    num_states = 0
                    num_arcs = 0
                    print("Error with arcs in ",r["fileC"])

            # print("Converting to OpenFST...")
            # oC = k2_to_openfst(C)
            # num_states= oC.num_states
            # num_arcs = ofstnumarcs(oC)

            # kaldifst.connect(oC)
            # num_arcs_conn = ofstnumarcs(oC)
            # num_states_conn = oC.num_states

            # print("Benchmarking...")
            times = timeit.repeat("k2.compose(k2.arc_sort(A),k2.arc_sort(B),treat_epsilons_specially=False)", globals=locals(),setup="import k2", number=1,repeat=_repetitions)

            results.append({"fileA":r["fileA"],
                            "fileB":r["fileB"],
                            "fileC":r["fileC"],
                            "mean":np.mean(times),"std":np.std(times),"min":np.min(times),"max":np.max(times), 
                            "nstates": num_states,
                            "narcs": num_arcs,
                            "conn_nstates": np.nan,
                            "conn_narcs":   np.nan
                            })
            

            pd.DataFrame(results).to_csv(output,index=False)

            

if __name__ == '__main__':
    fire.Fire(main)