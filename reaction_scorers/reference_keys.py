"""Reaction grouping in Python, with RDKit molecules and BLISS graph labels.

Fine groups retain changed bonds, functional groups and wildcard boundaries.
Coarse groups retain changed mapped atoms and attached unmapped fragments.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from rdkit import Chem
import igraph

from .reference import encode_reaction

SINGLE, DOUBLE, TRIPLE, ALTERNATE, LINK, MISSING = range(1, 7)
_VALENCES = {
    (symbol, charge): values
    for symbols, options in json.loads(
        Path(__file__).with_name("reference_valences.json").read_text()
    )
    for symbol in symbols
    for charge, values in options
}


@dataclass(frozen=True)
class Atom:
    element: int
    charge: int
    mapping: int
    side: int
    aromatic: bool = False  # membership in an sp2 cycle
    covered: bool = True

    @property
    def core(self):
        return self.element, self.charge


def pair(a, b):
    return min(a, b), max(a, b)


def reachable(seeds, neighbors):
    seen = set(seeds)
    pending = list(seen)
    while pending:
        for end in neighbors(pending.pop()):
            if end not in seen:
                seen.add(end)
                pending.append(end)
    return seen


def cycle_edges(atoms, edges):
    """Edges in cycles of the induced graph, computed by removing bridges."""
    adjacency = {i: [] for i in atoms}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].append(b)
            adjacency[b].append(a)
    discovery, low, bridges = {}, {}, set()

    def visit(node, parent=None):
        discovery[node] = low[node] = len(discovery)
        for end in adjacency[node]:
            if end == parent:
                continue
            if end not in discovery:
                visit(end, node)
                low[node] = min(low[node], low[end])
                if low[end] > discovery[node]:
                    bridges.add(pair(node, end))
            else:
                low[node] = min(low[node], discovery[end])

    for node in adjacency:
        if node not in discovery:
            visit(node)
    return {p for p in edges if set(p) <= atoms and p not in bridges}


def kekule_edges(molecule):
    """Assign aromatic single/double bonds by deterministic constraint propagation.
    """
    edges = {
        pair(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): (
            ALTERNATE
            if b.GetBondTypeAsDouble() == 1.5
            else int(b.GetBondTypeAsDouble())
        )
        for b in molecule.GetBonds()
    }
    adjacency = [[] for _ in molecule.GetAtoms()]
    for (a, b), kind in edges.items():
        if kind == ALTERNATE:
            adjacency[a].append(b)
            adjacency[b].append(a)
    for neighbors in adjacency:
        neighbors.sort()
    requires_double = []
    for i, atom in enumerate(molecule.GetAtoms()):
        minimum = atom.GetNumExplicitHs() + sum(
            1 if kind == ALTERNATE else kind for p, kind in edges.items() if i in p
        )
        options = _VALENCES.get((atom.GetSymbol(), atom.GetFormalCharge()))
        total = next((v for v in options if v >= minimum), None) if options else minimum
        requires_double.append(total is not None and total > minimum)
    singles = {
        p
        for p, kind in edges.items()
        if kind == ALTERNATE and not all(requires_double[i] for i in p)
    }
    doubles = set()

    def propagate_single(start, end):
        candidates = [
            other
            for other in adjacency[end]
            if other != start and pair(end, other) not in singles
        ]
        if not candidates:
            raise ValueError("Cannot assign aromatic bond orders")
        if len(candidates) == 1:
            other = candidates[0]
            p = pair(end, other)
            if p not in doubles:
                doubles.add(p)
                propagate_double(end, other)

    def propagate_double(start, end):
        for other in adjacency[end]:
            if other == start:
                continue
            p = pair(end, other)
            if p in doubles:
                raise ValueError("Two aromatic double bonds at one atom")
            if p not in singles:
                singles.add(p)
                propagate_single(end, other)

    for p, kind in sorted(edges.items()):
        if kind != ALTERNATE or p in singles or p in doubles:
            continue
        saved_singles, saved_doubles = singles.copy(), doubles.copy()
        try:
            doubles.add(p)
            propagate_double(*p)
            propagate_double(*reversed(p))
        except ValueError:
            singles, doubles = saved_singles, saved_doubles
            singles.add(p)
            propagate_single(*p)
            propagate_single(*reversed(p))
    return {
        p: DOUBLE if p in doubles else SINGLE if p in singles else kind
        for p, kind in edges.items()
    }


def alternate_edges(edges, size):
    """Find alternating single/double cycles up to 14 edges.

    This includes qualifying cycles in non-aromatic rings.
    Each ring atom must have exactly one double bond. Search order follows atom
    and bond order, and only the first qualifying cycle for each root is used.
    """
    adjacency = [[] for _ in range(size)]
    for (a, b), order in edges.items():
        adjacency[a].append((b, order))
        adjacency[b].append((a, order))
    for neighbors in adjacency:
        neighbors.sort()
    eligible = {
        i for i in range(size) if sum(order == DOUBLE for _, order in adjacency[i]) == 1
    }
    result = set()
    for root, order in sorted(edges.items()):
        if root in result or order not in (SINGLE, DOUBLE) or root[0] not in eligible:
            continue
        paths = [(root[0],)]
        found = None
        for depth in range(14):
            required = 3 - order if depth % 2 == 0 else order
            extended = []
            for path in paths:
                for end, kind in adjacency[path[-1]]:
                    if kind != required or end not in eligible:
                        continue
                    if pair(path[-1], end) == root:
                        found = path
                        break
                    if end not in path:
                        extended.append((*path, end))
                if found is not None:
                    break
            if found is not None:
                break
            paths = extended
            if not paths:
                break
        if found is not None:
            result.add(root)
            result.update(pair(a, b) for a, b in zip(found, found[1:]))
    return result


class Kernel:
    def __init__(self, atoms, edges):
        self.atoms = dict(enumerate(atoms)) if isinstance(atoms, list) else dict(atoms)
        self.edges = dict(edges)

    def neighbors(self, node, proper=False):
        for (a, b), order in self.edges.items():
            if proper and order in (LINK, MISSING):
                continue
            if a == node:
                yield b, order
            elif b == node:
                yield a, order

    def mirrors(self):
        return {
            node: end
            for (a, b), kind in self.edges.items()
            if kind == LINK
            for node, end in ((a, b), (b, a))
        }

    def relink(self):
        by_map = {}
        for i, atom in self.atoms.items():
            key = atom.side, atom.mapping
            if key in by_map:
                raise ValueError("Duplicate atom map on one reaction side")
            by_map[key] = i
        for (side, mapping), i in by_map.items():
            if side == 0 and (1, mapping) in by_map:
                self.edges[pair(i, by_map[1, mapping])] = LINK

    def redo_missing(self):
        self.edges = {p: kind for p, kind in self.edges.items() if kind != MISSING}
        mirrors = self.mirrors()
        missing = set()
        for (a, b), kind in self.edges.items():
            if kind not in (LINK, MISSING) and a in mirrors and b in mirrors:
                mirrored = pair(mirrors[a], mirrors[b])
                if mirrored not in self.edges:
                    missing.add(mirrored)
        self.edges.update((p, MISSING) for p in missing)

    def subset(self, selected):
        return Kernel(
            {i: a for i, a in self.atoms.items() if i in selected},
            {p: k for p, k in self.edges.items() if set(p) <= selected},
        )

    def key(self):
        """Canonical labeled graph, including cross-side links and boundaries.

        Bond kinds become labeled vertices; BLISS canonicalizes the resulting graph.
        """
        if not self.atoms:
            return ""
        indices = {i: position for position, i in enumerate(self.atoms)}
        labels = [
            (1, a.side, a.element, a.charge, int(a.aromatic), int(a.covered))
            for a in self.atoms.values()
        ]
        edges = []
        for (a, b), kind in self.edges.items():
            position = len(labels)
            labels.append((0, kind))
            edges.extend([(indices[a], position), (indices[b], position)])
        palette = sorted(set(labels))
        color_ids = {label: i for i, label in enumerate(palette)}
        graph = igraph.Graph(n=len(labels), edges=edges, directed=False)
        graph.vs["color"] = [color_ids[label] for label in labels]
        canonical = graph.permute_vertices(
            graph.canonical_permutation(color=graph.vs["color"])
        )
        ordered_edges = sorted(tuple(sorted(edge)) for edge in canonical.get_edgelist())
        return json.dumps(
            [palette, canonical.vs["color"], ordered_edges], separators=(",", ":")
        )


def reaction_kernel(request, simplify):
    atoms, edges = [], {}
    for side, smiles_list, maps_list in (
        (0, request["substrates"], request["substrates_to_map"]),
        (1, [request["product"]], [request["product_to_map"]]),
    ):
        for smiles, mappings in zip(smiles_list, maps_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() != len(mappings):
                raise ValueError("Invalid molecule or mapping length")
            offset = len(atoms)
            local = kekule_edges(mol)
            local.update(
                (p, ALTERNATE) for p in alternate_edges(local, mol.GetNumAtoms())
            )
            sp2 = set()
            for i, atom in enumerate(mol.GetAtoms()):
                bonds = [kind for p, kind in local.items() if i in p]
                valence = sum(1 if kind == ALTERNATE else kind for kind in bonds)
                valence += int(ALTERNATE in bonds)
                options = _VALENCES.get((atom.GetSymbol(), atom.GetFormalCharge()))
                total = (
                    next((v for v in options if v >= valence), None)
                    if options
                    else valence
                )
                if total is None:
                    raise ValueError("Invalid valence")
                if total - valence + len(bonds) < 4:
                    sp2.add(i)
            rings = cycle_edges(sp2, local)
            ring_atoms = {i for p in rings for i in p}
            atoms.extend(
                Atom(
                    a.GetAtomicNum(),
                    a.GetFormalCharge(),
                    mapping,
                    side,
                    i in ring_atoms,
                )
                for i, (a, mapping) in enumerate(zip(mol.GetAtoms(), mappings))
            )
            edges.update(
                (
                    pair(a + offset, b + offset),
                    ALTERNATE if simplify and (a, b) in rings else kind,
                )
                for (a, b), kind in local.items()
            )
    kernel = Kernel(atoms, edges)
    kernel.relink()
    return kernel


def coarse_kernel(kernel):
    mirrors = kernel.mirrors()

    def description(i):
        return (
            kernel.atoms[i].core,
            sorted(
                (kernel.atoms[end].mapping, kind, kernel.atoms[end].core)
                for end, kind in kernel.neighbors(i, proper=True)
            ),
        )

    changed = {
        i for i, other in mirrors.items() if description(i) != description(other)
    }
    selected = reachable(
        changed,
        lambda i: (
            end
            for end, _ in kernel.neighbors(i, proper=True)
            if end in changed or end not in mirrors
        ),
    )
    return kernel.subset(selected)


def fine_kernel(kernel):
    kernel.redo_missing()
    mirrors = kernel.mirrors()

    def mirror_pair(p):
        a, b = p
        return pair(mirrors[a], mirrors[b]) if a in mirrors and b in mirrors else None

    def changed_bond(p):
        other = mirror_pair(p)
        return other in kernel.edges and kernel.edges[p] != kernel.edges[other]

    changed = {
        i
        for i, other in mirrors.items()
        if kernel.atoms[i].core != kernel.atoms[other].core
    }
    for p, kind in kernel.edges.items():
        if kind != LINK and ((p[0] in mirrors) != (p[1] in mirrors) or changed_bond(p)):
            changed.update(p)
            changed.update(mirrors[i] for i in p if i in mirrors)
    aromatic = {i for i, atom in kernel.atoms.items() if atom.aromatic}
    aromatic_edges = cycle_edges(
        aromatic, {p for p, k in kernel.edges.items() if k not in (LINK, MISSING)}
    )
    # Add cross-side links and missing bonds between sp2-cycle atoms.
    # Missing bonds affect mirrored-bond classification.
    aromatic_edges.update(
        p
        for p, kind in kernel.edges.items()
        if kind in (LINK, MISSING) and set(p) <= aromatic
    )
    aromatic_atoms = {i for p in aromatic_edges for i in p}
    important = set()
    for i in changed:
        atom = kernel.atoms[i]
        bonds = list(kernel.neighbors(i, proper=True))
        if (
            atom.side == 0
            and atom.core == (6, 0)
            and i not in aromatic_atoms
            and all(
                kind == SINGLE and kernel.atoms[end].core == (6, 0)
                for end, kind in bonds
            )
        ):
            important.update(pair(i, end) for end, _ in bonds)
    for p, kind in kernel.edges.items():
        other = mirror_pair(p)
        is_aromatic = p in aromatic_edges or other in aromatic_edges
        interesting_end = any(kernel.atoms[i].core != (6, 0) for i in p)
        near_center = bool(set(p) & changed)
        if (
            kind == LINK
            or changed_bond(p)
            or other not in kernel.edges
            or (kind == DOUBLE and not is_aromatic)
            or kind == TRIPLE
            or (interesting_end and (near_center or not is_aromatic))
        ):
            important.add(p)
    important.update(
        other for p in list(important) if (other := mirror_pair(p)) in kernel.edges
    )
    core = reachable(
        changed,
        lambda i: (end for end, _ in kernel.neighbors(i) if pair(i, end) in important),
    )
    for p in set(kernel.edges) - important:
        del kernel.edges[p]
        for i in p:
            kernel.atoms[i] = replace(kernel.atoms[i], covered=False)
    kernel.edges = {
        p: kind
        for p, kind in kernel.edges.items()
        if kind != LINK or not all(kernel.atoms[i].covered for i in p)
    }
    kernel.redo_missing()
    boundary = reachable(
        (i for i, atom in kernel.atoms.items() if not atom.covered),
        lambda i: (end for end, _ in kernel.neighbors(i)),
    )
    return kernel.subset(core & boundary)


def exact_key(request):
    exact = []
    for smiles in (".".join(request["substrates"]), request["product"]):
        # Retain explicit stereo annotations even when molecular symmetry
        # makes their orientation redundant.
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        tags = [atom.GetChiralTag() for atom in mol.GetAtoms()]
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)  # Exact matching ignores isotopes.
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for atom, tag in zip(mol.GetAtoms(), tags):
            if (
                tag != Chem.ChiralType.CHI_UNSPECIFIED
                and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
            ):
                # Use an atom-map marker to retain stereo presence after sanitization.
                atom.SetAtomMapNum(1)
        exact.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ">>".join(exact)


def reaction_keys(substrates, product):
    request = encode_reaction(substrates, product)
    fine = fine_kernel(reaction_kernel(request, simplify=True)).key()
    coarse = coarse_kernel(reaction_kernel(request, simplify=False)).key()
    return {"fine": fine, "coarse": coarse, "exact": exact_key(request)}
