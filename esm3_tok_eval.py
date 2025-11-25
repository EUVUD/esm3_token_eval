from esm.sdk.api import ESMProtein
import numpy as np
import torch

from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL

from pymol import cmd

from Bio import pairwise2

def compute_rmsd(pdb1_name, pdb2_name):
    cmd.load("pdb_files/" + pdb1_name, "structure1")
    cmd.load("pdb_files/" + pdb2_name, "structure2")

    # Remove everything except protein
    cmd.remove("structure1 and not polymer.protein")
    cmd.remove("structure2 and not polymer.protein")

    # Align
    rmsd, _, _, _, _, _, _ = cmd.super("structure1", "structure2")

    cmd.save("cleaned_files/" + pdb1_name + "_cleaned.pdb", "structure1")
    cmd.save("cleaned_files/" + pdb2_name + "_cleaned.pdb", "structure2")
    
    # Clean up loaded structures
    cmd.delete("structure1")
    cmd.delete("structure2")

    return rmsd

def compute_l2_distance(pdb1_name, pdb2_name):
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL).to("cuda")

    protein1 = ESMProtein.from_pdb("cleaned_files/" + pdb1_name + "_cleaned.pdb")
    protein2 = ESMProtein.from_pdb("cleaned_files/" + pdb2_name + "_cleaned.pdb")
    protein1_tensor = model.encode(protein1)
    protein2_tensor = model.encode(protein2)

    struct_tokens1 = protein1_tensor.structure.to("cuda")
    struct_tokens2 = protein2_tensor.structure.to("cuda")

    struct_emb1 = model.encoder.structure_tokens_embed(struct_tokens1)
    struct_emb2 = model.encoder.structure_tokens_embed(struct_tokens2)

    seq1 = protein1.sequence
    seq2 = protein2.sequence

    aln = pairwise2.align.globalxx(seq1, seq2)[0]
    seq1_aln, seq2_aln = aln.seqA, aln.seqB

    idx1 = []
    idx2 = []

    p1 = p2 = 0

    for a, b in zip(seq1_aln, seq2_aln):
        if a != '-' and b != '-':   # positions exist in both proteins
            idx1.append(p1)
            idx2.append(p2)
        if a != '-':
            p1 += 1
        if b != '-':
            p2 += 1

    aligned_emb1 = struct_emb1[idx1]
    aligned_emb2 = struct_emb2[idx2]

    l2 = torch.norm(aligned_emb1 - aligned_emb2, dim=-1).mean()
    return l2.item()

def computel2_and_rmsd(pdb1_path, pdb2_path):
    rmsd = compute_rmsd(pdb1_path, pdb2_path)
    l2 = compute_l2_distance(pdb1_path, pdb2_path)
    return l2, rmsd

if __name__ == "__main__":
    pdb1 = "6Q1C.pdb"
    pdb2 = "6Q1D.pdb"

    l2, rmsd = computel2_and_rmsd(pdb1, pdb2)
    print(f"Average L2 Distance: {l2}")
    print(f"RMSD: {rmsd}")
