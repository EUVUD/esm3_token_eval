from pymol import cmd

cmd.load("pdb_files/6Q1C.pdb", "apo")
cmd.load("pdb_files/6Q1D.pdb", "holo")

# Remove everything except protein
cmd.remove("apo and not polymer.protein")
cmd.remove("holo and not polymer.protein")

# Align
rmsd, _, _, _, _, _, _ = cmd.super("apo", "holo")
print("RMSD:", rmsd)

cmd.save("apo_clean_aligned.pdb", "apo")
cmd.save("holo_clean_aligned.pdb", "holo")