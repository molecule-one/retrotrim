"""Syntheseus Retro* search with a public generator and optional reaction scorers."""

import argparse
import json
import math
import time
from pathlib import Path

from omegaconf import OmegaConf
from syntheseus.cli.search import BaseSearchConfig, SearchAlgorithmConfig, search_algorithm_config_to_kwargs
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.mol_inventory import SmilesListInventory

from .decision import DecisionRule
from .generators import DEFAULT_GENERATOR, DEFAULT_NUM_RESULTS, load_public_generator

NATIVE_SEARCH = SearchAlgorithmConfig()
NATIVE_RUN = BaseSearchConfig()


class ReactionFilter:
    def __init__(self, models_dir, device, scorers, rule, mapper=None):
        self.scorers = tuple(scorers)
        self.rule = rule
        self.mapper = mapper
        self.device = device
        self.cache = {}
        self.attempts = []
        self.models = {}
        self.index = None
        if set(scorers) & {"rp", "rgp"}:
            from .loading import load_rgp, load_rp

            for kind, loader in (("rp", load_rp), ("rgp", load_rgp)):
                if kind in scorers:
                    self.models[kind] = loader(Path(models_dir) / kind, device)
        if "rrs" in scorers:
            from .reference_sqlite import PythonReferenceIndex

            self.index = PythonReferenceIndex(Path(models_dir) / "rrs/reference.sqlite")

    def evaluate(self, reaction):
        key = reaction.reaction_smiles
        if key in self.cache:
            return self.cache[key]
        result = {}
        if not self.scorers:
            return True, result
        try:
            if self.mapper is None:
                from rxnmapper import RXNMapper

                self.mapper = RXNMapper()
                # Use the requested device for atom mapping.
                import torch

                self.mapper.device = torch.device(self.device)
                self.mapper.model.to(self.mapper.device)
            mapped = self.mapper.get_attention_guided_atom_maps([key])[0]["mapped_rxn"]
            substrates, product = mapped.split(">>")
            from .score import get_reaction

            prepared = get_reaction(substrates, product)
            for kind, scorer in self.models.items():
                result[kind + "_score"] = float(scorer.score([prepared])[0])
            if self.index is not None:
                result["reference_count"] = self.index.count(**prepared)["reference_count"]
            accepted = self.rule.accepts(result, self.scorers)
        except Exception as exc:
            accepted = False
            result["error"] = type(exc).__name__
        result = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in result.items()}
        self.cache[key] = accepted, result
        self.attempts.append({"reaction": key, "accepted": accepted, **result})
        return accepted, result

    def close(self):
        if self.index is not None:
            self.index.close()


class FilteredBackwardModel(BackwardReactionModel):
    def __init__(self, generator, reaction_filter, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.reaction_filter = reaction_filter

    def _get_reactions(self, inputs, num_results):
        batches = self.generator(inputs, num_results=num_results)
        output = []
        for reactions in batches:
            accepted = []
            for reaction in reactions:
                decision, scores = self.reaction_filter.evaluate(reaction)
                if decision:
                    reaction.metadata.update(scores)
                    accepted.append(reaction)
            output.append(accepted)
        return output


def search(target, inventory, generator, reaction_filter=None, *, num_results=DEFAULT_NUM_RESULTS, max_calls=500, max_depth=NATIVE_SEARCH.retro_star_config.max_expansion_depth, time_limit=NATIVE_SEARCH.time_limit_s, max_iterations=NATIVE_SEARCH.limit_iterations, stop_on_first_solution=True, max_routes=3):
    model = generator
    if reaction_filter is not None:
        model = FilteredBackwardModel(generator, reaction_filter, use_cache=True)
    model.default_num_results = num_results
    # Load native Retro* defaults and apply the requested search limits.
    config = OmegaConf.structured(SearchAlgorithmConfig())
    config.retro_star_config.max_expansion_depth = max_depth
    config.limit_reaction_model_calls = max_calls
    config.limit_iterations = max_iterations
    config.time_limit_s = float(time_limit)
    config.stop_on_first_solution = stop_on_first_solution
    algorithm = RetroStarSearch(reaction_model=model, mol_inventory=inventory,
                                **search_algorithm_config_to_kwargs(config))
    start = time.monotonic()
    graph, _ = algorithm.run_from_mol(Molecule(target))
    for node in graph.nodes():
        node.data["route_cost"] = node.data.get("retro_star_rxn_cost", 0.0)
    routes = []
    for nodes in iter_routes_cost_order(graph, max_routes=max_routes):
        routes.append([
            {"reaction": node.reaction.reaction_smiles, "metadata": dict(node.reaction.metadata)}
            for node in sorted(nodes, key=lambda n: (n.depth, str(n)))
            if isinstance(node, AndNode)
        ])
    return {
        "target": graph.root_node.mol.smiles,
        "has_solution": bool(graph.root_node.has_solution),
        "elapsed_seconds": time.monotonic() - start,
        "graph_nodes": len(graph),
        "reaction_model_calls": model.num_calls(),
        "search_settings": {
            "num_results": num_results, "limit_reaction_model_calls": max_calls,
            "limit_iterations": max_iterations, "max_expansion_depth": max_depth,
            "time_limit_s": time_limit, "stop_on_first_solution": stop_on_first_solution,
            "prevent_repeat_mol_in_trees": config.prevent_repeat_mol_in_trees,
        },
        "generator": "Syntheseus " + getattr(generator, "public_generator_name", generator.__class__.__name__.removesuffix("Model")),
        "generator_settings": getattr(generator, "public_generator_settings", {}),
        "scorers": list(reaction_filter.scorers) if reaction_filter else [],
        "routes": routes,
        "scored_reactions": reaction_filter.attempts if reaction_filter else [],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target")
    target.add_argument("--target-index", type=int, default=0, help="Zero-based index in the target file.")
    parser.add_argument("--targets", type=Path, default=Path("retrosynthesis_targets.txt"))
    parser.add_argument("--inventory", type=Path, required=True, help="Purchasable SMILES, one per line.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", choices=["rootaligned"], default=DEFAULT_GENERATOR)
    parser.add_argument("--num-augmentations", type=int, default=1, help="RootAligned SMILES augmentations; default matches the selected study profile.")
    parser.add_argument("--scorers", nargs="+", choices=["off", "rp", "rgp", "rrs"], default=["rp", "rgp", "rrs"])
    parser.add_argument("--decision", type=Path, help="Defaults to configs/decision_coverage90_rootaligned.json; tune for your targets and stock.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/syntheseus"))
    parser.add_argument("--device", help="Defaults to CUDA when available, otherwise CPU, as in native Syntheseus.")
    parser.add_argument("--num-results", type=int, default=DEFAULT_NUM_RESULTS)
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--max-iterations", type=int, default=NATIVE_SEARCH.limit_iterations)
    parser.add_argument("--max-depth", type=int, default=NATIVE_SEARCH.retro_star_config.max_expansion_depth)
    parser.add_argument("--time-limit", type=float, default=NATIVE_SEARCH.time_limit_s)
    parser.add_argument("--stop-on-first-solution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--canonicalize-inventory", action="store_true", default=NATIVE_RUN.canonicalize_inventory)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.num_augmentations < 1 or args.num_results < 1:
        parser.error("Augmentation and result counts must be positive.")
    if args.decision is None:
        args.decision = Path("configs/decision_coverage90_rootaligned.json")
    if args.device is None:
        import torch

        args.device = "cuda:0" if NATIVE_RUN.use_gpu and torch.cuda.is_available() else "cpu"
    if "off" in args.scorers and args.scorers != ["off"]:
        parser.error("Use 'off' alone, or list the scorers to enable.")
    target_smiles = args.target
    if target_smiles is None:
        targets = [line.strip() for line in args.targets.read_text().splitlines() if line.strip()]
        if not 0 <= args.target_index < len(targets):
            parser.error("Target index is outside the target file.")
        target_smiles = targets[args.target_index]
    reaction_filter = None
    if args.scorers != ["off"]:
        rule = DecisionRule.load(args.decision)
        reaction_filter = ReactionFilter(args.models_dir, args.device, args.scorers, rule)
    try:
        report = search(
            target_smiles,
            SmilesListInventory.load_from_file(args.inventory, canonicalize=args.canonicalize_inventory),
            load_public_generator(args.device, args.cache_dir, generator=args.generator,
                                  num_augmentations=args.num_augmentations, num_results=args.num_results),
            reaction_filter,
            num_results=args.num_results,
            max_calls=args.max_calls,
            max_depth=args.max_depth,
            time_limit=args.time_limit,
            max_iterations=args.max_iterations,
            stop_on_first_solution=args.stop_on_first_solution,
        )
    finally:
        if reaction_filter is not None:
            reaction_filter.close()
    report["decision_config"] = str(args.decision) if reaction_filter else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("routes", "scored_reactions")}, indent=2))


if __name__ == "__main__":
    main()
