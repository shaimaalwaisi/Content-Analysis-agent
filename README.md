# Content Analysis Agent

Annotates retail product images (Sony TV, mobile, audio) with marketing tags from a fixed
vocabulary, so content teams can tell at a glance what a product page is showing — and, by joining
tags to image views, which kinds of image actually earn attention.

Built as a small LangGraph agent shared by a CLI and a Streamlit UI.

## Quick start

```bash
pip install -r requirements.txt

# Works offline, no API key
python -m cli tag --input data/test/TV --provider mock
```

For real tagging, copy `.env.example` to `.env` and add an `ANTHROPIC_API_KEY`
([console.anthropic.com](https://console.anthropic.com/settings/keys)). `.env` is gitignored.

```bash
# Tag ten images; every row lands in results.sqlite3
python -m cli tag --input data/test --limit 10 --provider anthropic --few-shot 8 --workers 8

# ...and the Results tab shows them as one table
streamlit run app/streamlit_app.py
```

## Commands

| Command | What it does |
| --- | --- |
| `taxonomy` | Print the 43-tag controlled vocabulary |
| `tag --input DIR` | Tag a folder of images; `--output` writes JSON or CSV |
| `eval --train-dir DIR` | Score against the labels in the training filenames, versus baselines |
| `insights --from-sheet` | Rank tags by engagement using the metadata sheet |

Every run writes a timestamped JSON record to `results/` — the settings used, what came back, and
what it cost — so a run survives the terminal scrolling away. `--results-dir` moves them,
`--no-results` turns them off, and the app's Scores tab opens the newest `eval` record by default.

`tag` also writes one durable row per image into `results.sqlite3` (`--db` moves it, `--no-db` turns
it off). That is what the app's **Results** tab reads: image name, product, category, highlights,
price, views and the marketing tags, with the model's own reason for each tag underneath and a CSV
download. Price and views are joined from `meta_data.xlsx` when the table renders, so correcting the
sheet never means re-tagging.

Useful flags: `--provider` (`anthropic`, `openai`, `xai`, `groq`, `ollama`, `mock`), `--few-shot N`,
`--workers N`, `--enrich` (look products up for non-visual tags), `--no-memory`, `--no-db`,
`--report FILE`.

## Results

Claude Haiku 4.5 on the 8 labelled images:

| Run | micro-F1 | Jaccard | exact-match |
| --- | --- | --- | --- |
| prior top-3 baseline | 0.638 | 0.469 | 0.000 |
| zero-shot | 0.630 | 0.517 | 0.125 |
| **`--few-shot 8`** | **0.875** | **0.812** | **0.625** |

Few-shot is what matters: zero-shot loses to a baseline that ignores the image entirely. Eight
labelled images is a small sample and the model is not deterministic, so repeat runs land around
0.85–0.88 — directional, not a benchmark.

From the metadata sheet, tags on images that earn views (n=44): `awards` 1.68×, `product summary`
1.67×, `person` 1.47× the average, against `connectivity` 0.52×. Social proof and orienting shots
out-perform spec detail.

## Structure

```
agent/        the agent: graph, vlm, taxonomy, prompts, memory, retry, logging
tools/        capabilities it calls: search.py for non-visual tags, database.py for the results
evaluation/   measures the agent: quality.py (vs labels), runstats.py (vs nothing)
analysis/     measures the data: tags vs image views
cli/          terminal entry point, one module per subcommand
app/          Streamlit UI: tag an image, the results table, and a read-only scores tab
tests/        169 offline tests
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
* **analyze_image** — reason, act, check: the model proposes tags *and a reason for each*, `--enrich`
  adds what an image cannot show, and the vocabulary decides what survives. If nothing survives, or
  more tags were rejected than kept, the node runs once more with the rejected tags quoted back at
  the model. Capped at two passes, so an image can never cost more than two calls.
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
pytest        # 169 tests, ~1.7s, no network and no API key
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
  it, and views exist for just 44 of its 151 rows. `insights --from-sheet` reads tags from the
  sheet's own filenames; joining tagged test images to it matches nothing, and the command says so.
- Image files are gitignored; only the folder skeleton is committed.
