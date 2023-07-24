"""
This script is used to benchmark the composition operation in k2.
"""

#Name of the database to use
dbname = "fsadb_uw_composed"
repetitions = 500

import torch
torch.set_num_threads(1)
import k2
import numpy as np
import matplotlib.pyplot as plt
from kaldifst.utils import k2_to_openfst
import kaldifst
from pathlib import Path
import pandas as pd
import time
import timeit
from tqdm import tqdm


def float2bytesint(x):
    return np.frombuffer(bytes(np.array(x, dtype=np.float32)),dtype=np.int32)

def txtpath2k2fsa(filename):
    return k2.Fsa.from_openfst(open(filename).read(), num_aux_labels=1)

def ofstnumarcs(fsa):
    return sum([fsa.num_arcs(s) for s in range(fsa.num_states)])

print("Loading data...")


def main():
    df = pd.read_csv(f"{dbname}.csv")

    results = []
    output = f"{dbname}_k2_compbenchs.csv"
    if Path(output).exists():
        print("Loading previous results...")
        df_doned = pd.read_csv(output)
        results += df_doned.to_dict(orient="records")
    else:
        df_doned = None

    for i in tqdm(range(len(df))):
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
            print("Skipping ",r["fileC"])
            continue

        A = k2.arc_sort(A)
        B = k2.arc_sort(B)

        C = k2.compose(A,B)
        oC = k2_to_openfst(C)
        num_states= oC.num_states
        num_arcs = ofstnumarcs(oC)
        kaldifst.connect(oC)
        num_arcs_conn = ofstnumarcs(oC)
        num_states_conn = oC.num_states

        times = timeit.repeat("k2.compose(A,B)", globals=locals(), number=1,repeat=repetitions)

        results.append({"fileA":r["fileA"],
                        "fileB":r["fileB"],
                        "fileC":r["fileC"],
                        "mean":np.mean(times),"std":np.std(times),"min":np.min(times),"max":np.max(times), 
                        "nstates": num_states,
                        "narcs": num_arcs,
                        "conn_nstates": num_states_conn,
                        "conn_narcs":   num_arcs_conn
                        })

        pd.DataFrame(results).to_csv(output,index=False)


if __name__ == '__main__':
    fire.Fire(main)