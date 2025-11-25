import torch

emb1 = torch.load("6Q1C_structural_embedding.pt")   # shape: [L1, D]
emb2 = torch.load("6Q1D_structural_embedding.pt")   # shape: [L2, D]

# If sequences are same length:
l2_per_token = torch.norm(emb1 - emb2, dim=-1)
avg_l2 = l2_per_token.mean()

print("Average L2:", avg_l2.item())
