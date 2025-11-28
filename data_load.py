import pandas as pd

def load_pdb_pairs(csv_path):
    df = pd.read_csv(csv_path)
    pdb_pairs = [
        (f"{free}.pdb", f"{bound}.pdb")
        for free, bound in zip(df["structure free"], df["structure bound"])
    ]
    return pdb_pairs

if __name__ == "__main__":
    pdb_pairs = load_pdb_pairs("data/apoholo/apoholo_parsed_data.csv")
    print(f"Loaded {len(pdb_pairs)} PDB pairs.")
    print("First 5 pairs:", pdb_pairs[:5])