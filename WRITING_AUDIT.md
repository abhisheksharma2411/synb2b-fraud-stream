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

- sentences: 195
- mean length: 19.84 words
- standard deviation: 11.16 words  (gate: >= 11)
- min 3, max 57

| words | count |
|---|---|
| 0-4 | 6 |
| 5-9 | 39 |
| 10-14 | 30 |
| 15-19 | 23 |
| 20-24 | 29 |
| 25-29 | 32 |
| 30-34 | 19 |
| 35-39 | 8 |
| 40-44 | 3 |
| 45-49 | 3 |
| 50-54 | 2 |
| 55-59 | 1 |

## 2. Short and long sentences

- under 8 words: 31 (gate: >= 12)
- over 35 words: 14 (gate: >= 8)

  - short: "abstract Fraud scores are not decisions."
  - short: "An oracle propensity still leaves 0.086."
  - short: "A fraud model returns a number."
  - short: "Nobody pays a number."
  - short: "The policy chooses its own labels."
  - short: "This is not staleness."
  - long: "The best deployable one misses the ratio of true to estimated risk by 0.123, and every principled correction m..."
  - long: "Gibbs and Cand es replace the fixed quantile with an online recursion that adapts under shift and then under a..."
  - long: "Delayed and partial fraud feedback were documented a decade ago , and recent work formalises the limits delaye..."
  - long: "On that path the ledger shows only the product MATH , so MATH is recovered as a ratio: MATH fitted on matured ..."

## 3. Paragraph rhythm

- paragraphs: 45
- sentence counts in order: [11, 4, 3, 6, 3, 5, 3, 4, 3, 3, 4, 3, 7, 5, 6, 7, 6, 1, 6, 4, 3, 5, 1, 4, 3, 3, 4, 2, 3, 2, 3, 4, 10, 4, 8, 4, 6, 3, 2, 4, 8, 2, 5, 2, 6]
- runs of three consecutive equal counts: none

## 4. Banned vocabulary

None found. Zero occurrences of every banned word and phrase.

## 5. Punctuation and self-reference caps

- em-dashes: 0 (cap 5)
- semicolons: 3 (cap 8)
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

