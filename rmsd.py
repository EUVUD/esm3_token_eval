import MDAnalysis as mda
from MDAnalysis.analysis import align

u_ref = mda.Universe("pdb_files/6Q1C.pdb")
u_mob = mda.Universe("pdb_files/6Q1D.pdb")

ref = u_ref.select_atoms("protein and segid A")
mob = u_mob.select_atoms("protein and segid A")

# find common resids
common = sorted(set(ref.resids) & set(mob.resids))

sel_str = "protein and segid A and resid " + " ".join(str(r) for r in common) + " and name CA"
ref_sel = u_ref.select_atoms(sel_str)
mob_sel = u_mob.select_atoms(sel_str)

# must match
assert ref_sel.n_atoms == mob_sel.n_atoms

rmsd = align.alignto(mob_sel, ref_sel)[0]
print("Alignment RMSD:", rmsd)