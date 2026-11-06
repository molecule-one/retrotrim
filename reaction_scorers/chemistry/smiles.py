"""Canonical SMILES and deterministic multicanonical representations."""

from __future__ import annotations

import logging
from typing import Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

logger = logging.getLogger(__name__)


def complete_mappings(subs: Mol, prod: Mol) -> Tuple[Mol, Mol]:
    """Assign unused map numbers to unmapped atoms in both molecules, in place."""
    max_num = 0
    for mol in (subs, prod):
        all_map_nums = [a.GetAtomMapNum() for a in mol.GetAtoms()]
        if len(all_map_nums) > 0:
            max_num = max(max(all_map_nums), max_num)
    curr_num = max_num + 1
    for mol in (subs, prod):
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() <= 0:
                a.SetAtomMapNum(curr_num)
                curr_num += 1
    return (subs, prod)


def smiles_to_canonical(smi: str) -> str:
    """Return canonical SMILES, or an empty string if conversion fails."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        smi_can = Chem.MolToSmiles(mol, canonical=True)
        return smi_can
    except Exception as e:
        logger.warning(f"Exception while smiles_to_canonical on SMILES {smi}: {str(e)}")
        return ""


def smiles_to_multicanonical(smi: str, n: int = 0) -> str:
    """Root each component at atom `n`, reversing component order for odd `n`.

    Return an empty string if conversion fails. If `n` exceeds a component's
    atom count, use its last atom as the root.
    """
    assert isinstance(smi, str)
    try:
        new_smi = []
        smi = smiles_to_canonical(smi)
        for subsmi in smi.split("."):
            subsmi = smiles_to_canonical(subsmi)
            mol_subsmi = Chem.MolFromSmiles(subsmi)
            subsmi = Chem.MolToSmiles(
                mol_subsmi,
                canonical=True,
                rootedAtAtom=min(n, mol_subsmi.GetNumAtoms() - 1),
            )
            new_smi.append(subsmi)
        if n % 2 == 1:
            new_smi = new_smi[::-1]
        return ".".join(new_smi)
    except Exception as e:
        logger.warning(f"Exception while smiles_to_canonical on SMILES {smi}: {str(e)}")
        return ""
