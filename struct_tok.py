from esm.sdk.api import ESMProtein
import numpy as np
import torch

from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL

model = ESM3.from_pretrained(ESM3_OPEN_SMALL).to("cuda")
# model = ESM3.from_pretrained("EvolutionaryScale/esm3-sm-open-v1").to("cuda")

protein = ESMProtein.from_pdb("pdb_files/cleaned_6Q1D.pdb")

protein_tensor = model.encode(protein)

structure_tokens = protein_tensor.structure.to("cuda")

struct_emb = model.encoder.structure_tokens_embed(structure_tokens)

torch.save(struct_emb.cpu(), "6Q1D_structural_embedding.pt")



