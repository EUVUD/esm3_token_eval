import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import data_load
from Bio import pairwise2
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from esm.utils.constants.models import ESM3_OPEN_SMALL
from pymol import cmd, CmdException

APO_CHAIN_DIR = Path("data/apoholo/apo_chains")
HOLO_CHAIN_DIR = Path("data/apoholo/holo_chains")
CLEANED_DIR = Path("cleaned_files")
RMSD_FAILURE_THRESHOLD = 500.0
RESULTS_DIR = Path("results")
RMSD_CACHE_PATH = RESULTS_DIR / "rmsd_cache.csv"
RESULTS_CSV_PATH = RESULTS_DIR / "rmsd_l2_pairs.csv"
L2_CHUNK_SIZE = 100

def filter_existing_pairs(pdb_pairs):
    valid_pairs = []
    missing = []
    for apo, holo in pdb_pairs:
        apo_path = APO_CHAIN_DIR / apo
        holo_path = HOLO_CHAIN_DIR / holo
        if apo_path.exists() and holo_path.exists():
            valid_pairs.append((apo, holo))
        else:
            missing.append((apo, holo))
    if missing:
        print(f"Skipping {len(missing)} pairs with missing source PDB files.")
    return valid_pairs


def load_rmsd_cache(path: Path):
    cache = {}
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (row["apo_pdb"], row["holo_pdb"])
                rmsd_str = row.get("rmsd", "")
                cache[key] = float(rmsd_str) if rmsd_str else None
    return cache


def save_rmsd_cache(path: Path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["apo_pdb", "holo_pdb", "rmsd"])
        writer.writeheader()
        for (apo, holo), rmsd in cache.items():
            writer.writerow({
                "apo_pdb": apo,
                "holo_pdb": holo,
                "rmsd": "" if rmsd is None else rmsd,
            })


def load_l2_results(path: Path):
    results = {}
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (row["apo_pdb"], row["holo_pdb"])
                try:
                    rmsd_val = float(row["rmsd"])
                except ValueError:
                    continue
                try:
                    l2_val = float(row["l2"])
                except ValueError:
                    continue
                results[key] = {"rmsd": rmsd_val, "l2": l2_val}
    return results


def append_l2_results(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["apo_pdb", "holo_pdb", "rmsd", "l2"])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_rmsd(pdb1, pdb2):
    apo_path = APO_CHAIN_DIR / pdb1
    holo_path = HOLO_CHAIN_DIR / pdb2
    if not apo_path.exists() or not holo_path.exists():
        return None
    try:
        cmd.load(str(apo_path), "s1")
        cmd.load(str(holo_path), "s2")
        cmd.remove("s1 and not polymer.protein")
        cmd.remove("s2 and not polymer.protein")
        cmd.save(str(CLEANED_DIR / f"{pdb1}_cleaned.pdb"), "s1")
        cmd.save(str(CLEANED_DIR / f"{pdb2}_cleaned.pdb"), "s2")
        super_result = cmd.super("s1", "s2")
        if super_result is None:
            return None
        rmsd = super_result[0]
        aligned_atoms = super_result[1] if len(super_result) > 1 else None
        if not math.isfinite(rmsd) or rmsd > RMSD_FAILURE_THRESHOLD or (aligned_atoms is not None and aligned_atoms == 0):
            print(f"Skipping pair ({pdb1}, {pdb2}) due to non-converged RMSD ({rmsd}).")
            return None
        return rmsd
    except (CmdException, FileNotFoundError):
        return None
    finally:
        for obj in ("s1", "s2"):
            try:
                cmd.delete(obj)
            except CmdException:
                    pass


def compute_l2_for_pair(model, device, apo_name, holo_name):
    apo_clean = CLEANED_DIR / f"{apo_name}_cleaned.pdb"
    holo_clean = CLEANED_DIR / f"{holo_name}_cleaned.pdb"
    if not apo_clean.exists() or not holo_clean.exists():
        print(f"Skipping pair ({apo_name}, {holo_name}) during L2; cleaned file missing.")
        return None

    protein1 = ESMProtein.from_pdb(str(apo_clean))
    protein2 = ESMProtein.from_pdb(str(holo_clean))

    with torch.no_grad():
        enc1 = model.encode(protein1)
        struct_emb1 = model.encoder.structure_tokens_embed(enc1.structure.to(device)).cpu()
        del enc1
        if device.type == "cuda":
            torch.cuda.empty_cache()

        enc2 = model.encode(protein2)
        struct_emb2 = model.encoder.structure_tokens_embed(enc2.structure.to(device)).cpu()
        del enc2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seq1 = protein1.sequence
    seq2 = protein2.sequence

    aln = pairwise2.align.globalxx(seq1, seq2)[0]
    seq1_aln, seq2_aln = aln.seqA, aln.seqB

    idx1, idx2, p1_i, p2_i = [], [], 0, 0
    for a, b in zip(seq1_aln, seq2_aln):
        if a != '-' and b != '-':
            idx1.append(p1_i)
            idx2.append(p2_i)
        if a != '-':
            p1_i += 1
        if b != '-':
            p2_i += 1

    if not idx1:
        print(f"Skipping pair ({apo_name}, {holo_name}) during L2; no aligned residues.")
        return None

    aligned1 = struct_emb1[idx1]
    aligned2 = struct_emb2[idx2]
    l2_value = torch.norm(aligned1 - aligned2, dim=-1).mean().item()

    return l2_value


def compute_l2_in_chunks(model, device, ordered_pairs, rmsd_cache, chunk_size, results_path):
    existing = load_l2_results(results_path)
    pairs_to_process = [pair for pair in ordered_pairs if rmsd_cache.get(pair) is not None and pair not in existing]
    if not pairs_to_process:
        print("No new L2 computations required.")
        return

    total = len(pairs_to_process)
    for start in range(0, total, chunk_size):
        chunk = pairs_to_process[start:start + chunk_size]
        new_rows = []
        for apo, holo in chunk:
            l2_value = compute_l2_for_pair(model, device, apo, holo)
            if l2_value is None:
                continue
            new_rows.append({
                "apo_pdb": apo,
                "holo_pdb": holo,
                "rmsd": rmsd_cache[(apo, holo)],
                "l2": l2_value,
            })
        append_l2_results(results_path, new_rows)
        print(f"Completed L2 for {min(start + len(chunk), total)} / {total} pairs.")


if __name__ == "__main__":
    pdb_pairs = data_load.load_pdb_pairs("data/apoholo/apoholo_parsed_data.csv")

    #testrun size

    # test_pdb_pairs = pdb_pairs[:32]

    pdb_pairs = filter_existing_pairs(pdb_pairs)
    if not pdb_pairs:
        print("No valid PDB pairs to process; exiting.")
        sys.exit(0)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    rmsd_cache = load_rmsd_cache(RMSD_CACHE_PATH)

    valid_pairs = []
    new_rmsd_computations = 0
    for idx, pair in enumerate(pdb_pairs, start=1):
        cached_value = rmsd_cache.get(pair)
        if cached_value is None and pair in rmsd_cache:
            continue  # previously failed or missing
        if cached_value is None:
            rmsd_value = compute_rmsd(*pair)
            rmsd_cache[pair] = rmsd_value
            new_rmsd_computations += 1
            if new_rmsd_computations % 10 == 0:
                save_rmsd_cache(RMSD_CACHE_PATH, rmsd_cache)
        else:
            rmsd_value = cached_value

        if rmsd_value is not None:
            valid_pairs.append(pair)
        else:
            continue

    save_rmsd_cache(RMSD_CACHE_PATH, rmsd_cache)

    if not valid_pairs:
        print("No valid RMSD values available; exiting.")
        sys.exit(0)

    if new_rmsd_computations:
        print(f"Computed {new_rmsd_computations} new RMSD values.")
    skipped_pairs = len(pdb_pairs) - len(valid_pairs)
    if skipped_pairs:
        print(f"Skipping {skipped_pairs} pairs with missing or invalid RMSDs.")
    print(f"Proceeding with {len(valid_pairs)} pairs for L2 evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(device)
    model.eval()

    compute_l2_in_chunks(model, device, valid_pairs, rmsd_cache, L2_CHUNK_SIZE, RESULTS_CSV_PATH)

    results_map = load_l2_results(RESULTS_CSV_PATH)
    if not results_map:
        print("No L2 results available; skipping plot generation.")
        sys.exit(0)

    rmsd_values = [entry["rmsd"] for entry in results_map.values()]
    l2_values = [entry["l2"] for entry in results_map.values()]

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(rmsd_values, l2_values)
    plt.xlabel("RMSD")
    plt.ylabel("L2 distance")
    plt.title("RMSD vs L2 distance")
    plt.tight_layout()
    plt.savefig(output_dir / "rmsd_vs_l2.png", dpi=300)
    plt.close()

    print(f"Completed processing {len(results_map)} RMSD/L2 pairs. Results saved to {RESULTS_CSV_PATH}.")