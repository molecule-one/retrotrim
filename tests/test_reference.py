import math

import pytest

from reaction_scorers.reference import encode_molecule, reference_score


def test_singleton_reference_retains_a_nonzero_score():
    assert reference_score(0) == 0
    assert reference_score(1) == math.log(2)
    assert reference_score(2) > reference_score(1)


@pytest.mark.parametrize("count", [-1, 1.5, float("nan")])
def test_invalid_counts_are_rejected(count):
    with pytest.raises(ValueError):
        reference_score(count)


def test_reference_request_maps_follow_serialized_atom_order():
    pytest.importorskip("rdkit")
    smiles, mappings = encode_molecule("[OH:9][CH3:5]")
    assert smiles == "CO"
    assert mappings == [5, 9]


def test_reference_input_repairs_duplicate_and_missing_maps_consistently():
    from rdkit import Chem

    from reaction_scorers.reference import encode_reaction

    request = encode_reaction("[CH3:1][OH:1].[Cl-]", "[CH3:1][OH:1].[Na+]")
    substrate_maps = [value for maps in request["substrates_to_map"] for value in maps]
    product_maps = request["product_to_map"]
    assert len(set(substrate_maps)) == len(substrate_maps)
    assert len(set(product_maps)) == len(product_maps)
    assert set(substrate_maps) & set(product_maps) == {1, 2}
    assert all(value > 0 for value in substrate_maps + product_maps)
    product = Chem.MolFromSmiles(request["product"])
    carbon_map = next(
        value
        for atom, value in zip(product.GetAtoms(), product_maps)
        if atom.GetSymbol() == "C"
    )
    assert carbon_map == 1
