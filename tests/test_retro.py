import pytest

pytest.importorskip("syntheseus")
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.mol_inventory import SmilesListInventory

from reaction_scorers.decision import DecisionRule
from reaction_scorers.retro import ReactionFilter, search


class ToyGenerator(BackwardReactionModel):
    def _get_reactions(self, inputs, num_results):
        output = []
        for molecule in inputs:
            reactions = []
            if molecule.smiles == "CCO":
                for substrates, probability in (("C.CO", 0.7), ("CC.O", 0.2)):
                    reaction = SingleProductReaction.from_reaction_smiles(substrates + ">>CCO")
                    reaction.metadata["probability"] = probability
                    reactions.append(reaction)
            output.append(reactions[:num_results])
        return output


class ToyFilter:
    scorers = ("rp",)

    def __init__(self):
        self.attempts = []

    def evaluate(self, reaction):
        accepted = reaction.reactants_str == "CC.O"
        self.attempts.append({"reaction": reaction.reaction_smiles, "accepted": accepted})
        return accepted, {"rp_score": 0.9 if accepted else 0.1}


def test_real_syntheseus_search_can_enable_and_disable_reaction_filter():
    inventory = SmilesListInventory(["C", "CO", "CC", "O"])
    baseline = search("CCO", inventory, ToyGenerator(use_cache=True), num_results=2, max_calls=2)
    assert baseline["has_solution"]
    assert baseline["scorers"] == []
    assert len(baseline["routes"]) == 2
    reaction_filter = ToyFilter()
    filtered = search("CCO", inventory, ToyGenerator(use_cache=True), reaction_filter, num_results=2, max_calls=2)
    assert filtered["has_solution"]
    assert len(filtered["routes"]) == 1
    assert filtered["routes"][0][0]["reaction"] == "CC.O>>CCO"
    assert len(filtered["scored_reactions"]) == 2
    assert sum(x["accepted"] for x in filtered["scored_reactions"]) == 1


def test_disabled_filter_does_not_load_checkpoints_or_atom_mapper():
    class ExplodingMapper:
        def get_attention_guided_atom_maps(self, reactions):
            raise AssertionError("Disabled scorers must not invoke atom mapping.")

    reaction_filter = ReactionFilter("missing-model-directory", "cpu", (), DecisionRule(), mapper=ExplodingMapper())
    reaction = SingleProductReaction.from_reaction_smiles("CC.O>>CCO")
    assert reaction_filter.evaluate(reaction) == (True, {})
