import pandas as pd
from pathlib import Path
import fire

def main(path):
    dfs = []
    for p in Path(path).rglob("*.info"):
        dfx = pd.read_fwf(p,header=None).set_index(0)
        dfx.columns = [str(p)]
        dfs.append(dfx)
    pd.concat(dfs,axis=1).T.to_csv(f"{path}_info.csv")
   

if __name__ == "__main__":
    fire.Fire(main)