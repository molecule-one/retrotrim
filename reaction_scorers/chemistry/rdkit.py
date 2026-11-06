"""Reaction mapping and bond-change neighborhoods used by RP."""

from __future__ import annotations

from typing import List, Set, Tuple

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdchem import Mol


def is_mapped_reaction(substrates: str, product: str):
    """Check for an atom-map delimiter on each nonempty reaction side.
    """
    if (len(substrates) == 0 or ":" in substrates) and (
        len(product) == 0 or ":" in product
    ):
        return True
    else:
        return False


def get_changed_atoms(subs: Mol, prod: Mol) -> Set[int]:
    """Return map numbers whose bond neighborhoods differ between the sides."""
    ch_atoms = set()
    if subs is None and prod is None:
        return ch_atoms
    if subs is None:
        return set((a.GetAtomMapNum() for a in prod.GetAtoms()))
    if prod is None:
        return set((a.GetAtomMapNum() for a in subs.GetAtoms()))
    maps_subs = [a.GetAtomMapNum() for a in subs.GetAtoms()]
    maps_prod = [a.GetAtomMapNum() for a in prod.GetAtoms()]
    neigh_subs = {}
    neigh_prod = {}
    for mol, maps, neigh_dict in [
        (subs, maps_subs, neigh_subs),
        (prod, maps_prod, neigh_prod),
    ]:
        for bond in mol.GetBonds():
            map1, map2 = (maps[bond.GetBeginAtomIdx()], maps[bond.GetEndAtomIdx()])
            bond_type = str(bond.GetBondType())
            if map1 not in neigh_dict:
                neigh_dict[map1] = set()
            neigh_dict[map1].add((map2, bond_type))
            if map2 not in neigh_dict:
                neigh_dict[map2] = set()
            neigh_dict[map2].add((map1, bond_type))
    for m, bonds in neigh_subs.items():
        if m not in neigh_prod:
            ch_atoms.add(m)
        elif bonds != neigh_prod[m]:
            ch_atoms.add(m)
    for m, bonds in neigh_prod.items():
        if m not in neigh_subs:
            ch_atoms.add(m)
    return ch_atoms


def _remove_atom_mapping(mol):
    """Remove atom maps from an RDKit molecule in place."""
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    return mol


def get_changed_atoms_within_reaction_center(
    subs: Mol, prod: Mol, center_size
) -> Set[int]:
    """Return changed atom maps and neighbors up to `center_size` bonds away."""
    ch_atoms = get_changed_atoms(subs, prod)
    if center_size > 0:
        reactant_atoms = {atom.GetAtomMapNum(): atom for atom in subs.GetAtoms()}
        product_atoms = {atom.GetAtomMapNum(): atom for atom in prod.GetAtoms()}
        expanded_atoms = set(ch_atoms)
        for _ in range(center_size):
            for atom_map in list(expanded_atoms):
                atom = reactant_atoms.get(atom_map) or product_atoms.get(atom_map)
                expanded_atoms.update(
                    [neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()]
                )
        return expanded_atoms
    elif center_size == 0:
        return ch_atoms
    else:
        raise ValueError("Reaction center should be >= 0.")


def get_changed_atoms_indices_in_reaction_center(
    mapped_substrates: str, mapped_product: str, center_size=0, remove_mapping=False
) -> Tuple[str, List[int], List[int]]:
    """Return canonical reaction SMILES and center atom indices for each side.

    Indices follow the atom order in each returned SMILES. Set `remove_mapping`
    to strip atom maps from the returned reaction.
    """
    reactants = Chem.MolFromSmiles(mapped_substrates)
    products = Chem.MolFromSmiles(mapped_product)
    max_atom_num = max(
        [a.GetAtomMapNum() for a in reactants.GetAtoms()]
        + [a.GetAtomMapNum() for a in products.GetAtoms()]
    )
    for mol in [reactants, products]:
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() == 0:
                max_atom_num += 1
                a.SetAtomMapNum(max_atom_num)
    changed_atoms_map_nums = get_changed_atoms_within_reaction_center(
        reactants, products, center_size=center_size
    )
    reactant_atoms = {atom.GetAtomMapNum(): atom for atom in reactants.GetAtoms()}
    product_atoms = {atom.GetAtomMapNum(): atom for atom in products.GetAtoms()}
    if remove_mapping:
        reactants = _remove_atom_mapping(reactants)
        products = _remove_atom_mapping(products)
        Chem.SanitizeMol(reactants)
        Chem.SanitizeMol(products)
    reactants_canonical = Chem.MolToSmiles(reactants, canonical=True)
    products_canonical = Chem.MolToSmiles(products, canonical=True)
    last_digit_in_rdkit_string = (
        -2 if reactants.GetProp("_smilesAtomOutputOrder")[-2] == "," else -1
    )
    order_reactants = [
        int(x)
        for x in reactants.GetProp("_smilesAtomOutputOrder")[
            1:last_digit_in_rdkit_string
        ].split(",")
    ]
    order_products = [
        int(x)
        for x in products.GetProp("_smilesAtomOutputOrder")[
            1:last_digit_in_rdkit_string
        ].split(",")
    ]
    changed_atom_indices_substrates = [
        order_reactants.index(atom.GetIdx())
        for atom_map, atom in reactant_atoms.items()
        if atom_map in changed_atoms_map_nums
    ]
    changed_atom_indices_product = [
        order_products.index(atom.GetIdx())
        for atom_map, atom in product_atoms.items()
        if atom_map in changed_atoms_map_nums
    ]
    reaction_canonical = reactants_canonical + ">>" + products_canonical
    return (
        reaction_canonical,
        sorted(changed_atom_indices_substrates),
        sorted(changed_atom_indices_product),
    )


def mapped_smiles_to_complete_mapped(subs: str, prod: str) -> Tuple[str, str]:
    """Assign map numbers to unmapped atoms and return the two SMILES strings."""
    assert is_mapped_reaction(subs, prod)
    sub_mol = MolFromSmiles(subs)
    prod_mol = MolFromSmiles(prod)
    if sub_mol is None or prod_mol is None:
        return (None, None)
    sub_mol, prod_mol = complete_mappings(sub_mol, prod_mol)
    return (MolToSmiles(sub_mol), MolToSmiles(prod_mol))


def complete_mappings(subs: Mol, prod: Mol) -> Tuple[Mol, Mol]:
    """Assign unused map numbers to unmapped atoms in both molecules, in place."""
    max_num = 0
    for mol in (subs, prod):
        max_num = max(max((a.GetAtomMapNum() for a in mol.GetAtoms())), max_num)
    curr_num = max_num + 1
    for mol in (subs, prod):
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() <= 0:
                a.SetAtomMapNum(curr_num)
                curr_num += 1
    return (subs, prod)


def smiles_to_unmapped(smi: str, sanitize: bool = False) -> str:
    """Remove atom maps and return canonical SMILES; return an empty string if parsing fails.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    for a in mol.GetAtoms():
        a.ClearProp("molAtomMapNumber")
    if sanitize:
        Chem.SanitizeMol(mol)
    smi_unmapped = Chem.MolToSmiles(mol, canonical=True)
    return smi_unmapped
