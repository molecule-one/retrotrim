# Reaction benchmark

`reactions.csv` contains 500 human-annotated reactions:

| Column | Meaning |
|---|---|
| `substrates` | Dot-separated, atom-mapped reactant SMILES |
| `product` | Atom-mapped product SMILES |
| `category` | Plausibility category: Safe bet, Worthwhile, Rather not, or Nonsense |
| `issue` | Chemical issue category, or No problem |

For the paper's binary evaluation, Safe bet is positive; Rather not and
Nonsense are negative. Worthwhile is excluded. The 500 unique reactions contain 394
Safe bet, 59 Rather not, 42 Nonsense, and 5 Worthwhile reactions, yielding
495 evaluated reactions (394 positive and 101 negative).

Selection was score-informed, balancing class proportions, score distributions,
rank contributions, and issue categories. Metrics below are for this selected sample, excluding Worthwhile reactions.

| Scorer | ROC-AUC | PR-AUC (average precision) |
|---|---:|---:|
| RP | 0.73441 | 0.91485 |
| RGP | 0.71435 | 0.90385 |
| RRS | 0.71800 | 0.89045 |

[selection.json](selection.json) records the CSV checksum, schema, category
counts, and aggregate metrics. Follow the [scoring instructions](../README.md#score-and-evaluate-reactions)
to compute and evaluate predictions.
