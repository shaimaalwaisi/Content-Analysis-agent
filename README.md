# Content Analysis Agent

Annotates retail product images (Sony TV, mobile, audio) with marketing tags from a fixed
vocabulary, so content teams can tell at a glance what a product page is showing — and, by joining
tags to image views, which kinds of image actually earn attention.

Built as a small LangGraph agent shared by a CLI and a Streamlit UI.

## Quick start

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add an `ANTHROPIC_API_KEY`
([console.anthropic.com](https://console.anthropic.com/settings/keys)). `.env` is gitignored.
Tagging calls Claude, so the key is required — the only thing that runs without one is `pytest`.

```bash
# Tag ten images; every row lands in results.sqlite3
python -m cli tag --input data/test --limit 10 --few-shot 8 --workers 8

# ...and the Results tab shows them as one table
streamlit run app/streamlit_app.py
```

## Commands

| Command | What it does |
| --- | --- |
| `taxonomy` | Print the 43-tag controlled vocabulary |
| `tag --input DIR` | Tag a folder of images; `--output` writes JSON or CSV |
| `eval --train-dir DIR` | Score against the labels in the training filenames, versus baselines |
| `tag --consistency` | Tag twice with different examples and score the agreement (no labels needed) |
| `insights --from-sheet` | Rank tags by engagement using the metadata sheet |

Every run writes a timestamped JSON record to `results/` — the settings used, what came back, and
what it cost — so a run survives the terminal scrolling away. `--results-dir` moves them,
`--no-results` turns them off, and the app's Metrics tab opens the newest `tag` record by default.

`tag` also writes one durable row per image into `results.sqlite3` (`--db` moves it, `--no-db` turns
it off). That is what the app's **Results** tab reads: image name, product, category, description,
specs, highlights and the marketing tags, with the model's own reason for each tag underneath and a
CSV download. Price and views are joined from `meta_data.xlsx` when the table renders — and hidden
when it has no row for the batch, which is the case for every test image (see the data notes).

Useful flags: `--few-shot N`,
`--workers N`, `--enrich` (look products up for non-visual tags), `--no-memory`, `--no-db`.

## Metrics

**Two for quality**, scored by `eval` against the tags in the training filenames. Scoring is
leave-one-out: with `--few-shot`, the image being scored is dropped from its own example list, so
the model is never shown the answer it is being asked for.

| Metric | What it answers |
| --- | --- |
| Micro-F1 | Pooled over every (image, tag) decision: are the tags right? |
| Macro-F1 | Every tag weighted equally: are the *rare* tags right too? A model that only ever predicts the two commonest tags scores well on micro and badly here. |

Both are printed next to two model-free baselines, because on this data "always guess the most
common tags" is a high floor — `physical design` appears in every label.

**Three for the workflow**, reported by every `tag` run. None of them needs labels, so all three
work on the unlabelled test set and on live traffic:

| Metric | What it answers |
| --- | --- |
| Task success rate | Of the images we were asked to tag, how many came back with usable tags? An image that raised and an image that returned an empty list both count as failures. |
| Cost per task | Input and output tokens priced at Anthropic's published rate, divided by tasks. A model with no rate card reports blank, never `$0.00`. |
| Latency per action | Wall time per action — image encode, model call, search call — as p50/p95. Per action, because a re-prompted image makes two model calls and a per-task figure hides which one is slow. |

**One for reliability**, opt-in with `tag --consistency`. It tags every image a second time with a
different draw of few-shot examples and scores the overlap between the two answers (Jaccard, per
image and averaged). No labels needed, so unlike micro-F1 it works on the 107 test images — but it
measures *steadiness, not correctness*: a model can be repeatably wrong. It costs a second model
call per image, which is why it is off by default.

The three workflow metrics appear in the app's **Metrics** tab, scoped to your own session. The
Results table adds a per-image **How sure?** column — confidence, not correctness: it flags rows
that were re-prompted, carry unexplained tags, or came back with a single tag, so a reviewer knows
which images to open first.

## Results

Claude Haiku 4.5, `--few-shot 8`, leave-one-out over the 8 labelled training images:

| Run | micro-F1 | macro-F1 |
| --- | --- | --- |
| constant baseline (always `physical design`) | 0.516 | 0.100 |
| prior top-3 baseline | 0.638 | 0.221 |
| **agent** | **0.857** | **0.760** |

Reproduce with `python -m cli eval --train-dir data/train --few-shot 8`.

Read it as directional, not a benchmark: n=8, all one phone model, and the model is not
deterministic, so repeat runs move by a few points. What it does establish is that the agent clears
both label-prior baselines on both metrics — a model that could not would be adding nothing over
"guess the commonest tags".

From the metadata sheet, tags on images that earn views (n=44): `awards` 1.68×, `product summary`
1.67×, `person` 1.47× the average, against `connectivity` 0.52×. Social proof and orienting shots
out-perform spec detail.

## Structure

```
agent/        the agent: graph, vlm, taxonomy, prompts, memory, enrichment, retry, logging
tools/        capabilities it calls: search.py for non-visual tags, database.py for the results
evaluation/   quality.py (2 vs labels), runstats.py (3 workflow), consistency.py (1)
data/         the images, meta_data.xlsx, and metadata.py: tags joined to image views
cli/          terminal entry point, one module per subcommand
app/          Streamlit UI: tag up to 10 uploads, the results table, the metrics tab
tests/        118 offline tests
```

Dependencies point inward: the outer folders import `agent`; it imports none of them at runtime.

The agent is three nodes, one branch and one loop:

```
prepare ─(memory hit)──────────────────────────────┐
        └(miss)─► analyze_image ─(good answer)─────┴─► persist ─► END
                       ▲     │
                       └─────┘ weak answer: ask once more
```

* **prepare** — encode the image, infer `Category`/`Model` from its folder path, ask memory.
* **analyze_image** — reason, act, check: the model proposes tags *and a reason for each*, plus the
  four things the vocabulary cannot say — category, model name, a one-line description, and the
  specs it can literally read in the image — `--enrich` adds what an image cannot show, and the
  vocabulary decides which tags survive. If nothing survives, or more tags were rejected than kept,
  the node runs once more with the rejected tags quoted back at the model. Capped at two passes, so
  an image can never cost more than two calls, and the details ride along in the same request.
* **persist** — the memory row that lets an identical request skip the model, and the durable
  results row the table reads. A memory hit routes through here too, so ten images are always ten
  rows.

Validation is the guardrail: anything outside the vocabulary is dropped, so neither the model nor a
search tool can invent a tag — and the feedback sent back into the loop only ever names tags that
were *rejected*, so it cannot steer the model towards a particular answer. Swapping providers,
editing `agent/taxonomy.json`, or changing the graph reaches every entry point, because they all go
through `build_graph`.

## Tests

```bash
pytest        # 118 tests, ~1.3s, no network and no API key
```

Fixtures synthesise their own images, so a fresh clone can run them despite the dataset being
gitignored. The suite is mutation-checked: breaking the few-shot leakage guard, the vocabulary
filter, media-type detection, the reasoning loop's bound, or the one-row-per-image rule each fails a
specific named test.

## Notes on the data

- Only **8 labelled training images** exist, all of one phone and almost all tagged
  `physical design`, so evaluation measures angle vocabulary and says nothing about
  `feature graphics` or `usage scene`.
- `meta_data.xlsx` describes the **training** images only — none of the 107 test images appear in
  it, and views exist for just 44 of its 151 rows. Its 12 product models do not include any of the
  test set's (it has `XPERIA1MK4`; the test folder has `XPERIA1MK5`), so no price or view count
  exists to join for a test image, and the results table hides those two columns rather than
  showing a wall of blanks. `insights --from-sheet` reads tags from the sheet's own filenames;
  joining tagged test images to it matches nothing, and the command says so.
- Image files are gitignored; only the folder skeleton is committed.
