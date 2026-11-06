"""Extract local reaction templates and enumerate competing product sites."""

import itertools
import logging
import re

from rdkit import Chem
from rdkit.Chem import AllChem

from .rdkit import get_changed_atoms_indices_in_reaction_center, smiles_to_unmapped
from .smiles import smiles_to_canonical
from .snapping import make_snapped_mol_by_idxs

logger = logging.getLogger(__name__)


def _fix_aromaticity_in_smarts(smarts, molecules):
    """Preserve aromatic element symbols in templates, e.g. #6 becomes c."""
    aromatic = {
        str(atom.GetAtomMapNum()): atom.GetIsAromatic()
        for molecule in molecules
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum()
    }
    symbols = {"#6": "c", "#7": "n", "#8": "o", "#16": "s"}

    def replace_atom(match):
        element, rest = match.groups()
        number = re.search(r":\d+", rest).group()[1:]
        if element in symbols and aromatic.get(number, False):
            element = symbols[element]
        return f"[{element}{rest}]"

    return re.sub(r"\[(\#\d+)([^\]]*)\]", replace_atom, smarts)


def extract_reaction_template_smarts(reaction, radius, whole_rings=False):
    """Extract the mapped substrate/product SMARTS within a bond-change radius."""
    canonical, substrate_atoms, product_atoms = (
        get_changed_atoms_indices_in_reaction_center(
            reaction["substrates"],
            reaction["product"],
            center_size=radius,
            remove_mapping=False,
        )
    )
    if not substrate_atoms:
        return None
    substrates, product = canonical.split(">>")
    snapped_product = make_snapped_mol_by_idxs(
        product, product_atoms, whole_rings=whole_rings
    )
    snapped_substrates = make_snapped_mol_by_idxs(
        substrates, substrate_atoms, whole_rings=whole_rings
    )
    originals = [Chem.MolFromSmiles(reaction[key]) for key in ("substrates", "product")]
    for snapped, original in zip((snapped_substrates, snapped_product), originals):
        for atom in snapped.GetAtoms():
            original_atom = original.GetAtomWithIdx(atom.GetIntProp("orig_atom_idx"))
            atom.SetNumExplicitHs(original_atom.GetNumExplicitHs())
    template = (
        Chem.MolToSmarts(snapped_substrates) + ">>" + Chem.MolToSmarts(snapped_product)
    )
    return _fix_aromaticity_in_smarts(template, originals)


def generate_alternative_template_reactions(reaction, rc_radius=2, whole_rings=False):
    """Apply a local template at other sites, excluding the observed product."""
    original_product = smiles_to_unmapped(smiles_to_canonical(reaction["product"]))
    smarts = extract_reaction_template_smarts(reaction, rc_radius, whole_rings)
    if smarts is None:
        return []
    template = AllChem.ReactionFromSmarts(smarts)
    substrates = [Chem.MolFromSmiles(s) for s in reaction["substrates"].split(".")]
    if len(substrates) != len(smarts.split(">>")[0].split(".")):
        logger.debug("Template and reaction have different numbers of substrates.")
        return []
    products = []
    for permutation in itertools.permutations(substrates):
        for candidate in template.RunReactants(permutation):
            try:
                products.append(".".join(Chem.MolToSmiles(mol) for mol in candidate))
            except Exception:
                logger.debug("Could not convert an alternative product to SMILES.")
    products = [smiles_to_canonical(product) for product in products]
    products = list(
        set(
            product
            for product in products
            if smiles_to_unmapped(product) != original_product and product
        )
    )
    return [
        {"substrates": reaction["substrates"], "product": product}
        for product in products
    ]
