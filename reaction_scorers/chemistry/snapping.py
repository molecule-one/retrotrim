"""Crop reactions around changed atoms, retaining rings and optional hydrogen caps."""

from __future__ import annotations

from typing import List, Set

import rdkit
from rdkit import Chem

from reaction_scorers.chemistry.rdkit import (
    get_changed_atoms_indices_in_reaction_center,
    is_mapped_reaction,
)


def _add_ring_atoms(mol: Chem.Mol, atom_idx: int, atom_set: Set[int]):
    """Add every atom sharing a ring with `atom_idx` to `atom_set`."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            atom_set.update(ring)


def make_snapped_reaction(
    mapped_substrates: str,
    mapped_product: str,
    rc_radius: int = 3,
    remove_radicals: bool = False,
) -> tuple[Chem.Mol | None, Chem.Mol | None]:
    """Return substrate/product fragments around changed atoms, including rings."""
    assert is_mapped_reaction(mapped_substrates, mapped_product)
    _, atom_ids_sub, atom_ids_prod = get_changed_atoms_indices_in_reaction_center(
        mapped_substrates=mapped_substrates,
        mapped_product=mapped_product,
        center_size=rc_radius,
        remove_mapping=False,
    )
    snapped_prod_mol = make_snapped_mol_by_idxs(
        mapped_product, atom_ids_prod, True, remove_radicals=remove_radicals
    )
    snapped_subs_mol = make_snapped_mol_by_idxs(
        mapped_substrates, atom_ids_sub, True, remove_radicals=remove_radicals
    )
    return (snapped_subs_mol, snapped_prod_mol)


def make_snapped_mol_by_idxs(
    product: str,
    idxs: List[int],
    whole_rings: bool = False,
    remove_radicals: bool = False,
) -> Chem.Mol:
    """Keep selected atom indices, optionally retaining their complete rings."""
    mol = Chem.MolFromSmiles(product)
    for at in mol.GetAtoms():
        at.SetIntProp("orig_atom_idx", at.GetIdx())
    em = Chem.EditableMol(mol)
    inside_range = set(idxs)
    if whole_rings:
        for atom_idx in list(inside_range):
            if mol.GetAtomWithIdx(atom_idx).IsInRing():
                _add_ring_atoms(mol, atom_idx, inside_range)
    outside_range = [j for j in range(len(mol.GetAtoms())) if j not in inside_range]
    for i in sorted(outside_range, reverse=True):
        em.RemoveAtom(i)
    mol = em.GetMol()
    Chem.GetSymmSSSR(mol)
    mol.UpdatePropertyCache(strict=True)
    if remove_radicals:
        mol = rdkit.Chem.MolFromSmiles(rdkit.Chem.MolToSmiles(mol))
        if mol is not None:
            for i in range(mol.GetNumAtoms()):
                if mol.GetAtomWithIdx(i).GetNumRadicalElectrons() >= 1:
                    mol.GetAtomWithIdx(i).SetNumExplicitHs(
                        mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                    )
                    mol.GetAtomWithIdx(i).SetNumRadicalElectrons(0)
    return mol
