# Content Analysis Agent

Tags retail product images (Sony TV, mobile, audio) with marketing terminology from a fixed
43-word vocabulary, so a content team can ask which shots show the camera — and, by joining tags
to image views, which kinds of shot actually earn attention.

A small LangGraph agent on Claude Haiku 4.5, shared by a CLI and a Streamlit UI.

![The agent tagging ten test images end to end](agent-run.gif)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

Tagging calls Claude, so the key is required. `pytest` is the only thing that runs without one.

```bash
python -m cli tag --input data/test --limit 10 --few-shot 8 --workers 8
streamlit run app/streamlit_app.py     # the Results tab shows what that wrote
```

## How it works

![prepare, analyze image, persist — with a memory shortcut and a retry loop](agent-diagram.png)

- **prepare** — encode the image, read `Category`/`Model` off its folder path, ask memory. A hit
  skips the model entirely.
- **analyze image** — the model proposes tags *and a reason for each*, plus what the vocabulary
  cannot say: category, model name, a one-line description, and any specs legible in the image.
  `--enrich` adds a web lookup for tags a photograph cannot show (awards, benchmark, energy
  rating). If nothing survives validation, or more tags were rejected than kept, the node runs
  once more with the rejected tags quoted back. Capped at two passes.
- **persist** — one memory row so an identical request is free next time, and one durable results
  row per image. A memory hit routes through here too, so ten images are always ten rows.

Validation is the guardrail: anything outside `agent/taxonomy.json` is dropped, so neither the
model nor the search tool can invent a tag. The feedback sent into the retry names only the
*rejected* tags, never the right answer, so a second attempt stays independent evidence.

All three entry points compile the same graph through `build_graph`, so a change reaches every one.

## Fetching from Sony

Images reach the agent two ways, and both end as a `Category/Model` folder: upload them, or fetch
them off a product page.

![one Sony page: twelve images fetched, described, stored and tagged](fetch-run.gif)

```bash
python -m cli fetch --url https://www.sony.co.uk/headphones/products/wf-1000xm6 --tag --few-shot 8
```

That run is real: twelve product shots off Sony's CDN, each described from the file itself, written
to the `images` table, then tagged — with the model's own reason beside every tag.

![a Sony page through three tools into the folder the tagging agent reads](ingestion-diagram.png)

- **fetch** (`tools/scraper.py`) — reads the page, finds the product shots, downloads the ones it
  has not seen before. Honours `robots.txt`, backs off on a 429, and skips any picture whose
  perceptual hash is already in the database, so a second run costs nothing.
- **describe** (`tools/imagemeta.py`) — width, height, format, EXIF, `sha256`, and the perceptual
  hash the deduplication turns on. Read off the file: no model is asked anything.
- **store** (`tools/database.py`) — one row per picture in `results.sqlite3 · images`, keyed by
  that hash. The tags land next door, in `taggings`.
- **tag** (`agent/graph.py`) — the agent, unchanged. It reads the folder the fetch wrote, exactly
  as it reads a folder you uploaded, which is why the fetch route needed no new node and no
  classifier: the page URL names the category, and the folder carries it.

`--scraper mock` draws its own pictures through the identical code path, so the route can be tried
with no network and no key. If Sony's edge refuses your machine — it answers 403 to a script on
most networks — save the page in your browser and pass it with `--html`, or drop it into the
**Fetch from Sony** tab; the images themselves come from a CDN that does answer.

## Docker

```bash
cp .env.example .env                 # only tagging needs the key
docker compose up app                # the UI on http://localhost:8501
docker compose run --rm cli tag --input data/test --limit 10 --few-shot 8
docker compose run --rm tests        # the suite, no key required
```

One image serves all three: the `app` service runs Streamlit, `cli` runs any subcommand, `tests`
runs `pytest`. Your images and run records stay on the host (`./data`, `./results`); the two SQLite
files live on a named volume, so rebuilding the image keeps the tagging history and the memory that
makes a repeat run free. The image runs as uid 10001 — if it cannot write `./data`, run
`UID=$(id -u) GID=$(id -g) docker compose up app`.

## Commands

| Command | What it does |
| --- | --- |
| `taxonomy` | Print the 43-tag controlled vocabulary |
| `fetch --url URL` | Download a Sony page's product images into `Category/Model` folders and record what each file is; `--tag` tags them in the same run |
| `fetch --url URL --html FILE` | The same, reading a page you saved from your browser — for networks Sony's edge refuses |
| `tag --input DIR` | Tag a folder; `--output` writes JSON or CSV |
| `eval --train-dir DIR` | Score against the labels in the training filenames, versus baselines |
| `tag --consistency` | Tag twice with different examples and score the agreement (no labels needed) |
| `insights --from-sheet` | Rank tags by engagement using the metadata sheet |

Useful flags: `--few-shot N`, `--workers N`, `--enrich`, `--no-memory`, `--no-db`.

Every run writes a timestamped JSON record to `results/` — the settings, what came back, and what
it cost. `tag` also writes one row per image into `results.sqlite3`, which is what the app's
**Results** tab reads: tags, the model's reason for each, description, specs and highlights, with
a CSV download.

## Results

### Tag quality, against human labels

Claude Haiku 4.5, `--few-shot 8`, scored leave-one-out over the 8 labelled training images — the
image being scored is dropped from its own example list, so the model never sees its own answer.

| Run | micro-F1 | macro-F1 |
| --- | --- | --- |
| constant baseline (always `physical design`) | 0.516 | 0.100 |
| prior top-3 baseline | 0.638 | 0.221 |
| **agent** | **0.745** | **0.555** |

Re-run it with `python -m cli eval --train-dir data/train --few-shot 8`, which writes the full
record to `results/`.

Read it as directional, not a benchmark: n=8, all one phone model, and the model is not
deterministic, so repeat runs move by several points. What it does establish is that the agent
clears both label-prior baselines on both metrics — on this data "always guess the commonest tags"
is a high floor, because `physical design` appears in every label.

Micro-F1 asks whether the tags are right; macro-F1 weights every tag equally, so it asks whether
the *rare* tags are right too. A model that only ever predicts the two commonest tags scores well
on the first and badly on the second.

### Metrics that need no labels

Micro- and macro-F1 need ground truth, which exists for 8 images. These four measure the agent
itself, so they work on the 107 unlabelled test images and on live traffic. Every `tag` and `eval`
run reports them, and they appear in the app's **Metrics** tab.

| Metric | What it answers | The run above |
| --- | --- | --- |
| Task success rate | Of the images we were asked to tag, how many came back with usable tags? An image that raised and an image that returned an empty list both count as failures. | 1.000 — 8/8, 0 errors, 0 retries |
| Cost per task | Input and output tokens at Anthropic's published rate, divided by tasks. A model with no rate card reports blank, never `$0.00`. | $0.0068 per image, $0.054 for the run |
| Latency per action | Wall time per *action* — encode, model call, search call — as p50/p95. Per action, because a re-prompted image makes two model calls and a per-task figure hides which one is slow. | model call 1.75 s p50, 2.57 s p95 |
| Self-consistency | Tag everything a second time with a different draw of examples and score the overlap. Steadiness, not correctness: a model can be repeatably wrong. | opt-in — `tag --consistency`, doubles the cost |

From the metadata sheet, tags on images that earn views (n=44): `awards` 1.68×, `product summary`
1.67×, `person` 1.47× the average, against `connectivity` 0.52×. Social proof and orienting shots
out-perform spec detail.

## Layout

```
agent/        the agent: graph, vlm, taxonomy, prompts, memory, enrichment, retry, logging
tools/        what it calls: scraper.py (fetch a page), imagemeta.py (read a file),
              search.py (non-visual tags), database.py (the results and images tables)
evaluation/   quality.py (vs labels), runstats.py (workflow), consistency.py
data/         images, meta_data.xlsx, and metadata.py: tags joined to image views
cli/          terminal entry point, one module per subcommand
app/          Streamlit UI: tag uploads, fetch from Sony, the results table, the metrics tab
tests/        157 offline tests
```

Dependencies point inward: the outer folders import `agent`; it imports none of them at runtime.

## Tests

```bash
pytest        # 157 tests, ~1.7 s, no network and no API key
```

Fixtures synthesise their own images, so a fresh clone runs them despite the dataset being
gitignored. The fetch route is covered the same way: `MockScraper` draws its own pictures, and the
live scraper is tested through its pure parts — URL resolution, the CDN shapes it accepts, the
naming rules, and what it does with a page it is refused.

## Notes on the data

- Only **8 labelled training images** exist, all of one phone and almost all tagged
  `physical design`, so evaluation measures angle vocabulary and says little about
  `feature graphics` or `usage scene`.
- `meta_data.xlsx` describes the **training** images only. None of the 107 test images appear in
  it, and views exist for 44 of its 151 rows. Its 12 product models do not overlap the test set's
  (it has `XPERIA1MK4`; the test folder has `XPERIA1MK5`), so no price or view count can be joined
  to a test image, and the results table hides those two columns rather than showing blanks.
- Image files are gitignored; only the folder skeleton is committed.
