import pytest

pytest.importorskip("rdkit")
pytest.importorskip("igraph")

from reaction_scorers.reference import encode_reaction
from reaction_scorers.reference_keys import ALTERNATE, reaction_kernel, reaction_keys
from reaction_scorers.reference_sqlite import PythonReferenceIndex, build_index


def test_keys_ignore_atom_order_and_map_numbers():
    first = reaction_keys("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH:2]=[O:3]")
    reordered = reaction_keys("[OH:9][CH2:8][CH3:7]", "[O:9]=[CH:8][CH3:7]")
    assert first == reordered


def test_coarse_groups_omit_unchanged_distant_carbon_chain():
    short = reaction_keys("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH:2]=[O:3]")
    long = reaction_keys("[CH3:4][CH2:1][CH2:2][OH:3]", "[CH3:4][CH2:1][CH:2]=[O:3]")
    assert short["coarse"] == long["coarse"]
    assert short["exact"] != long["exact"]


def test_sp2_cycle_normalization_does_not_use_rdkit_aromaticity():
    from rdkit import Chem

    substrate = "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6][CH:7]=[CH:8]1"
    product = "[CH2:1]1[CH2:2][CH:3]=[CH:4][CH:5]=[CH:6][CH:7]=[CH:8]1"
    assert not any(a.GetIsAromatic() for a in Chem.MolFromSmiles(substrate).GetAtoms())
    kernel = reaction_kernel(encode_reaction(substrate, product), simplify=True)
    assert all(a.aromatic for a in kernel.atoms.values() if a.side == 0)
    assert all(
        kind == ALTERNATE
        for (a, b), kind in kernel.edges.items()
        if kernel.atoms[a].side == kernel.atoms[b].side == 0
    )


def test_aromatic_flag_does_not_replace_an_explicit_triple_bond():
    from rdkit import Chem
    from reaction_scorers.reference_keys import TRIPLE, kekule_edges

    molecule = Chem.MolFromSmiles("c1ccccc#1")
    triple = next(b for b in molecule.GetBonds() if b.GetBondTypeAsDouble() == 3)
    # RDKit can mark this bond aromatic while retaining its explicit TRIPLE type.
    assert triple.GetIsAromatic()
    edges = kekule_edges(molecule)
    assert (
        edges[tuple(sorted((triple.GetBeginAtomIdx(), triple.GetEndAtomIdx())))]
        == TRIPLE
    )


def test_exact_keys_retain_stereo_but_ignore_isotopes():
    plain = reaction_keys("[CH3:1][OH:2]", "[CH3:1][OH:2]")
    isotope = reaction_keys("[13CH3:1][OH:2]", "[13CH3:1][OH:2]")
    assert plain == isotope
    left = reaction_keys("F[C@H](Cl)Br", "F[C@H](Cl)Br")
    right = reaction_keys("F[C@@H](Cl)Br", "F[C@@H](Cl)Br")
    assert left["exact"] != right["exact"]


def test_exact_keys_preserve_explicit_pseudoasymmetric_stereo_tags():
    from reaction_scorers.reference_keys import exact_key

    unspecified = {"substrates": ["OC[C@@H](O)C(O)[C@H](O)CO"], "product": "CO"}
    explicit = {"substrates": ["OC[C@@H](O)[C@H](O)[C@H](O)CO"], "product": "CO"}
    reversed_tag = {"substrates": ["OC[C@@H](O)[C@@H](O)[C@H](O)CO"], "product": "CO"}
    assert exact_key(unspecified) != exact_key(explicit)
    assert exact_key(explicit) == exact_key(reversed_tag)


def test_symmetric_kernel_key_is_invariant_to_atom_and_bond_order():
    import random
    from reaction_scorers.reference_keys import (
        Atom,
        Kernel,
        DOUBLE,
        LINK,
        MISSING,
        SINGLE,
        pair,
    )

    atoms = [
        Atom(6, 0, i + 1, side, covered=False) for side in (0, 1) for i in range(4)
    ]
    edges = {
        (0, 1): DOUBLE,
        (1, 2): MISSING,
        (2, 3): DOUBLE,
        (0, 3): MISSING,
        (4, 5): SINGLE,
        (5, 6): SINGLE,
        (6, 7): SINGLE,
        (4, 7): SINGLE,
        **{(i, i + 4): LINK for i in range(4)},
    }
    expected = Kernel(atoms, edges).key()
    rng = random.Random(20200101)
    for _ in range(64):
        ordering = list(range(8))
        rng.shuffle(ordering)
        inverse = {old: new for new, old in enumerate(ordering)}
        shuffled_edges = list(edges.items())
        rng.shuffle(shuffled_edges)
        permuted = Kernel(
            [atoms[i] for i in ordering],
            {pair(inverse[a], inverse[b]): kind for (a, b), kind in shuffled_edges},
        )
        assert permuted.key() == expected


def test_index_union_duplicates_and_last_exact_match(tmp_path):
    keys = [
        {"fine": "a", "coarse": "a", "exact": "x"},
        {"fine": "a", "coarse": "b", "exact": "y"},
        {"fine": "b", "coarse": "a", "exact": "z"},
        {"fine": "a", "coarse": "a", "exact": "x"},
        {"fine": "", "coarse": "b", "exact": "x"},
    ]
    rows = [{"substrates": str(i), "product": ""} for i in range(len(keys))]
    path = tmp_path / "index.sqlite"
    build_index(rows, path, key_function=lambda sub, prod: keys[int(sub)])
    with PythonReferenceIndex(path) as index:
        result = index.count_keys(keys[0])
        assert result == {
            "reference_count": 5,
            "has_exact_match": True,
            "details": {"rtype_reference_count": 3, "minimal_reference_count": 3},
        }
        assert index.count_keys(keys[4])["details"]["rtype_reference_count"] == 0


def test_failed_build_preserves_existing_output_and_removes_temporary(tmp_path):
    path = tmp_path / "index.sqlite"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_index([], path)
    assert path.read_bytes() == b"existing"
    assert not path.with_name("index.sqlite.building").exists()
    path.unlink()
    with pytest.raises(ValueError, match="Expected"):
        build_index([], path, expected_rows=1)
    assert not path.exists()
    assert not path.with_name("index.sqlite.building").exists()


def test_explicit_source_exclusions_keep_source_and_indexed_counts_distinct(tmp_path):
    rows = [
        {"substrates": "[CH3:1][OH:2]", "product": "[CH3:1][OH:2]"},
        {"substrates": "intentionally excluded invalid input", "product": ""},
        {"substrates": "[CH3:1][OH:2]", "product": "[CH3:1][OH:2]"},
    ]
    path = tmp_path / "index.sqlite"
    report = build_index(rows, path, expected_rows=3, excluded_rows=[1])
    assert report["source_rows"] == 3
    assert report["indexed_rows"] == 2
    assert report["skipped_rows"] == 1
    with PythonReferenceIndex(path) as index:
        result = index.count(**rows[0])
        assert result["reference_count"] == 2
        assert result["has_exact_match"]


def test_parallel_build_preserves_row_order_exclusions_and_exact_overwrites(tmp_path):
    first = {"substrates": "[CH4:1]", "product": "[CH3:1][OH:2]"}
    remapped = {"substrates": "[CH4:1]", "product": "[CH3:2][OH:1]"}
    rows = [first, remapped] * 130
    serial, parallel = tmp_path / "serial.sqlite", tmp_path / "parallel.sqlite"
    for path, workers in ((serial, 1), (parallel, 2)):
        report = build_index(
            rows, path, expected_rows=260, excluded_rows=[127, 129], workers=workers
        )
        assert report["indexed_rows"] == 258
    with PythonReferenceIndex(serial) as left, PythonReferenceIndex(parallel) as right:
        for reaction in (first, remapped):
            assert left.count(**reaction) == right.count(**reaction)
        assert left.count(**first)["reference_count"] == 131
        assert left.count(**remapped)["reference_count"] == 128
