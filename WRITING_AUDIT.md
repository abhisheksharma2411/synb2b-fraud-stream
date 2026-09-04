# WRITING_AUDIT.md

Output of `tools/audit_prose.py`. Every gate below is a hard threshold from the
paper's own writing rules. None of them were relaxed to obtain a pass.

**Overall: PASS**

| gate | result |
|---|---|
| sigma >= 11 | PASS |
| short sentences (<8w) >= 12 | PASS |
| long sentences (>35w) >= 8 | PASS |
| no three consecutive equal paragraph lengths | PASS |
| banned vocabulary == 0 | PASS |
| em-dashes <= 5 | PASS |
| semicolons <= 8 | PASS |
| we propose/present/introduce <= 3 | PASS |
| conjunction openers >= 6 | PASS |
| no First/Second/Third/Finally ladder | PASS |
| every prose number traces to results.json | PASS |
| pages (named) <= 6 | PASS |
| pages (anon) <= 6 | PASS |

## 1. Sentence-length distribution

- sentences: 187
- mean length: 20.78 words
- standard deviation: 11.16 words  (gate: >= 11)
- min 2, max 53

| words | count |
|---|---|
| 0-4 | 9 |
| 5-9 | 23 |
| 10-14 | 33 |
| 15-19 | 27 |
| 20-24 | 31 |
| 25-29 | 23 |
| 30-34 | 18 |
| 35-39 | 14 |
| 40-44 | 4 |
| 45-49 | 2 |
| 50-54 | 3 |

## 2. Short and long sentences

- under 8 words: 19 (gate: >= 12)
- over 35 words: 22 (gate: >= 8)

  - short: "abstract Fraud scores are not decisions."
  - short: "A fraud model returns a number."
  - short: "Nobody pays a number."
  - short: "The policy chooses its own labels."
  - short: "This is not staleness."
  - short: "Four things follow."
  - long: "Downstream of the score a payment is released, parked for dual-control approval, or dropped into an analyst's ..."
  - long: "An item released on the auto-allow path is labelled only if something surfaces, whether a chargeback represent..."
  - long: "Concurrent and independent work by Deng asks when fraud operations may authorize automation at all, under shar..."
  - long: "It also treats label arrival as exogenous, which is the assumption this paper spends its length attacking, and..."

## 3. Paragraph rhythm

- paragraphs: 44
- sentence counts in order: [10, 4, 2, 6, 3, 5, 4, 6, 3, 4, 4, 3, 5, 6, 6, 1, 3, 5, 4, 5, 2, 4, 2, 3, 4, 2, 4, 3, 4, 4, 7, 5, 5, 6, 5, 7, 6, 6, 4, 3, 4, 3, 4, 1]
- runs of three consecutive equal counts: none

## 4. Banned vocabulary

None found. Zero occurrences of every banned word and phrase.

## 5. Punctuation and self-reference caps

- em-dashes: 0 (cap 5)
- semicolons: 6 (cap 8)
- "we propose/present/introduce": 0 (cap 3)

## 6. Conjunction openers

- sentences opening with But/So/Yet/And/Then: 8 (gate: >= 6)

## 7. Numeric traceability

Every numeric literal in the prose matches a value in `results/results.json`,
`results/drift_events.json`, `src/t13/drift_params.json`, or the structural
allowlist in `tools/prose_numbers_allowlist.json` (64 entries,
each carrying its justification).

## 8. Enumeration ladders

None.

## 9. Page count

- named: 6 pages (ok)  (cap 6)
- anon: 6 pages (ok)  (cap 6)

## Iterations to reach a clean run

4 audit passes in total: the first found the failures below, and
3 rewrites cleared them. No threshold in
`tools/audit_prose.py` was changed at any point, and no gate was marked passed
on a near miss. Every failure was fixed in the paper.

1. First audit: sigma, short/long counts, banned vocabulary, openers and numeric traceability passed. Three consecutive paragraphs of six sentences, then three of four, failed the rhythm gate; semicolons stood at 9 against a cap of 8.
2. Merged one sentence in the exploration-ablation paragraph and one in the production-risk paragraph to break both runs, and turned three semicolons into a sentence break, a conjunction and a comma. Prose gates went clean; the named build was 7 pages, with 19 words of bibliography spilling onto page 7.
3. Cut the duplicated repository URL from the acknowledgment, which already appears under the toggle in Section V. Both builds reached 6 pages. Section 6.6's cut list was not needed, so Fig. 2 stayed.
4. Replaced two loose fractions with measured values after cross-checking them against results.json: 'a tenth of the base rate' became a qualitative statement, and 'a fifth of the analyst budget' became 0.120.

