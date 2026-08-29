# Content Analysis Agent

Annotates retail product images (Sony Mobile, TV, Video & Sound) with marketing tags drawn from a
fixed, controlled vocabulary. A single agent — built as a small LangGraph state machine — is shared
by the command line, the batch pipeline, the evaluation harness, and a Streamlit demo UI.

Every predicted tag is validated against the taxonomy before it is returned, so the output can never
contain an invented label.

## Quick start

```bash
pip install -r requirements.txt

# Works offline, no API key required
python -m content_analysis_agent.cli tag --input data/test/TV --provider mock
```

To use a real vision model, create a `.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys) (or
[platform.openai.com](https://platform.openai.com/api-keys) for the OpenAI provider). `.env` is
gitignored and never committed.

## Usage

### Command line

```bash
# Print the controlled vocabulary (43 tags)
python -m content_analysis_agent.cli taxonomy

# Tag a folder of images, recursively
python -m content_analysis_agent.cli tag --input data/test --provider anthropic \
    --output results.json          # a .csv extension writes CSV instead

# Score the agent against the labels baked into the training filenames
python -m content_analysis_agent.cli eval --train-dir data/train \
    --provider anthropic --sample 30 --report metrics.json
```

Useful flags: `--limit N` caps how many images are tagged, `--model` overrides the model id,
`--provider` selects `anthropic` (default), `openai`, or `mock`, and `--few-shot N` prepends N
labelled training images as worked examples (see [Few-shot prompting](#few-shot-prompting)).

Results look like:

```json
{
  "path": "data/test/TV/XR-65A95K/amazon_co_uk_B09XBQBXS2_Main_Carousel_1_2023-06-19.jpg",
  "category": "TV",
  "model": "XR-65A95K",
  "tags": ["physical design", "front angle"]
}
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Upload a single image, pick a provider in the sidebar, optionally raise the **Few-shot examples**
slider, and press **Tag image**.

Uploaded files land in a temp path, so the agent cannot infer the product from the folder name the
way the CLI does. Fill in the **Product context** box (e.g. `Category: TV, Model: XR-65A95K`) to give
the model that hint. Choose the `mock` provider to try the interface with no API key.

## How it works

```
taxonomy.json  ->  prompts  ->  VLM client  ->  graph  ->  pipeline / evaluate  ->  CLI / UI
```

The agent itself is five nodes, with a branch on memory:

```
load_image -> recall -+-(hit)-------------------------------> END
                      |
                      +-(miss)-> tag_image -> validate_tags -> remember -> END
```

| Module | Responsibility |
| --- | --- |
| `taxonomy.py` / `taxonomy.json` | The controlled vocabulary as a two-level General → Specific hierarchy, editable without touching code. |
| `prompts.py` | Every LLM prompt, in one place, so wording can be tuned centrally. |
| `vlm.py` | Swappable providers behind one `VLMClient` protocol: Anthropic, OpenAI, and an offline mock. |
| `graph.py` | The LangGraph agent, including the validation step that drops out-of-vocabulary tags. |
| `pipeline.py` | Batch-tags a folder, tolerating per-image failures so one bad file cannot abort a run. |
| `labels.py` | Parses the ground-truth tags encoded in training filenames. |
| `memory.py` | Persistent tag memory (SQLite), so identical requests skip the model. |
| `logconf.py` | Structured JSON-lines logging. |
| `retry.py` | Exponential backoff for transient provider failures. |
| `metadata.py` | Joins tags to the metadata sheet and ranks tags by engagement. |
| `tools.py` | Search tools for the enrich step, mock and Claude web search. |
| `fewshot.py` | Turns the labelled training images into few-shot demonstrations. |

Evaluation lives outside the package, in a top-level `evaluation/` folder alongside `app/`:
`quality.py` scores tagging against labels, `runstats.py` measures agent behaviour. Dependencies
point inward — `evaluation` imports `content_analysis_agent`, never the reverse. `graph` and
`pipeline` accept a `RunStats` under `TYPE_CHECKING` only, so the core has no runtime dependency on
the layer that measures it and imports cleanly on its own.

```
content_analysis_agent/   the agent and everything it needs to produce tags
evaluation/               measures the agent: quality.py, runstats.py
app/                      Streamlit UI
```

Because every entry point goes through `build_graph`, the graph is the extension point: inserting a
node between `tag_image` and `validate_tags` — say, an enrichment step that looks up non-visual tags
such as `awards` or `benchmark` — reaches the CLI, the pipeline, evaluation, and the UI at once.

### Providers

| `--provider` | Default model | Key |
| --- | --- | --- |
| `anthropic` | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `xai` | `grok-2-vision-1212` | `XAI_API_KEY` |
| `groq` | `qwen/qwen3.8-27b` | `GROQ_API_KEY` |
| `ollama` | `llama3.2-vision` | none — local server |
| `mock` | — | none. Deterministic, offline; for wiring and demos only, **never** for measuring accuracy |

Everything below `anthropic` speaks the OpenAI wire format, so one client class covers them all —
they differ only by base URL, key variable, and default model. That makes a free or local model a
one-flag experiment:

```bash
python -m content_analysis_agent.cli eval --train-dir data/train --provider groq --few-shot 4
```

Model ids move quickly on these services; pass `--model` if a default has been retired. Most of
Groq's catalogue is text-only — `openai/gpt-oss-120b` and the `compound` models reject image content
outright — so the default is the vision model verified to accept it.

**Providers cap images per request.** `qwen3.8-27b` allows three, so `--few-shot 2` (two examples
plus the target) is its ceiling; asking for more returns a 400 for every image. `eval` prints a loud
warning when images fail rather than reporting the resulting zeros as a score. Because the
memory cache key includes the model id, switching providers never returns another model's tags, and
the baseline table in `eval` shows immediately whether a cheaper model still clears the floor.

SDK imports are lazy, so the mock path runs without either provider package installed.

## Few-shot prompting

The 8 labelled training images are too few to train a vision model, but they are well suited to
few-shot prompting: `--few-shot N` prepends N of them to the request as worked
`image → tags` demonstrations, which costs nothing to prepare and needs no extra annotation.

```bash
python -m content_analysis_agent.cli tag --input data/test --provider anthropic --few-shot 4
python -m content_analysis_agent.cli eval --train-dir data/train --provider anthropic --few-shot 4
```

During evaluation the image being scored is always removed from its own example set, so examples and
test items can share one folder without inflating the score.

Note that all 8 labelled images are of a single phone and are tagged almost entirely under
`physical design`, so they teach angle vocabulary well and say nothing about `feature graphics`,
`usage scene`, or the other categories. Examples are not free either: each one is re-sent with every
request, so `--few-shot 8` multiplies image tokens per call by nine.

## Memory

The agent remembers what it has already decided. Before calling the model, the `recall` node looks
the request up in a small SQLite store; on a hit it routes straight to the end and no model call is
made. After a successful tagging run, `remember` writes the result back.

Entries are keyed by a hash of everything that could change the answer — the image bytes, the model
id, the product context, and the few-shot examples — so switching provider, editing the context, or
changing `--few-shot` all correctly miss rather than returning a stale answer.

```bash
# Second run over the same folder reuses everything and calls no model
python -m content_analysis_agent.cli tag --input data/test --provider anthropic
# Memory: 107 hit(s), 0 miss(es) (100% reused), 107 entries in .agent_memory.sqlite3

python -m content_analysis_agent.cli tag --input data/test --no-memory   # force fresh calls
python -m content_analysis_agent.cli tag --input data/test --memory /tmp/other.sqlite3
```

Memory is on by default and stored in `.agent_memory.sqlite3` (gitignored). The Streamlit sidebar has
a matching **Reuse remembered tags** checkbox, and the UI says when a result came from memory.

## Image downscaling

Images longer than 1024px on their longest side are downscaled and re-encoded as JPEG before upload
(`vlm.encode_image`). Vision models bill by pixel area rather than file size, so on this dataset the
resize cuts roughly **51% of image tokens** — about $0.75 to $0.36 per full 107-image run at Sonnet
input pricing — with no loss of tagging detail at this resolution. Pass `max_dim=0` to send originals.

## Running at scale

**Parallelism.** Tagging is network-bound, so `--workers N` tags N images at once. On this dataset
that is roughly a 6× wall-clock saving at 8 workers, and output order always matches folder order —
work is submitted in order and the futures are read in order, so parallelism never reshuffles
results or progress output.

```bash
python -m content_analysis_agent.cli tag --input data/test --provider anthropic --workers 8
```

**Retries.** Transient provider failures (429, 408/409, 5xx, connection and timeout errors) are
retried with exponential backoff and jitter. Non-transient failures — a malformed request, a bad API
key — are raised immediately rather than retried, since they fail identically every time. Without
this, one rate-limit response would silently become an empty tag list for that image.

**Logging.** Diagnostics are emitted as JSON lines, one object per event, ready for `jq` or a log
shipper. Human progress output stays on stdout.

```bash
python -m content_analysis_agent.cli --log-level INFO --log-file run.log tag --input data/test
```

```json
{"ts": "...", "level": "INFO",    "event": "image_tagged", "image": "...", "n_tags": 2, "ms": 84, "cached": false}
{"ts": "...", "level": "WARNING", "event": "out_of_vocab_tags", "image": "...", "dropped": ["sparkly"]}
{"ts": "...", "level": "INFO",    "event": "run_complete", "images": 107, "workers": 8, "seconds": 41.2}
```

`out_of_vocab_tags` is the metric worth alerting on: the validation step silently drops predictions
outside the taxonomy, so a rising drop rate is the earliest signal of prompt or taxonomy drift.

## Which tags earn attention

Tagging answers *what is in this image*. The brief opens with a different question — marketing wants
to know "what content is popular and engaging" — and supplies `meta_data.xslx` with file names,
categories, models, prices and **image views**. The `insights` command answers it.

```bash
# Rank tags by engagement using the sheet's own labelled file names
python -m content_analysis_agent.cli insights --from-sheet --metadata data/meta_data.xlsx --min-support 3
```

```
tag               images  mean views      lift
awards                 4        48,000.0   1.68x
product summary        3        47,666.7   1.67x
person                 3        42,000.0   1.47x
front angle            7        37,285.7   1.30x
...
connectivity           3        15,000.0   0.52x

Overall mean views: 28,590.9
```

**Lift** is the tag's mean metric over the overall mean: above 1.0 means images carrying that tag
out-perform average. On the supplied data, social proof (`awards`, `person`) and orienting shots
(`product summary`, `front angle`) draw roughly 1.3–1.7× the average, while spec-detail imagery
(`connectivity`, `controls`, `application`) draws about half. `--min-support N` hides tags too rare
to read anything into, and `--metric` ranks by any numeric column (`price` instead of `views`).

### Reading the supplied sheet honestly

Three properties of `meta_data.xslx` shape what can be concluded from it:

- It describes the **labelled training images only** — every one of its 151 rows has a bracketed,
  tag-encoded name. It contains **no rows for the test images**, so tagging the test set and joining
  it to views matches nothing. `insights --input ...` reports that plainly rather than printing an
  empty table.
- **Views are present for only 44 of 151 rows**, covering three products (one per category). Every
  figure above rests on those 44.
- **File names repeat across products** — they are tag lists, not unique ids — so the join key is
  (category, model, file name), not the name alone.

Column headers are matched fuzzily, so the supplied `Name` / `Image views` / `Product price` all
resolve, as do variants like `File Name`, `image_filename` and `views_30d`; `.xlsx` and `.csv` both
load. `--results results.json` reuses an earlier tagging run instead of paying for it twice, and
`--synthetic PATH` writes a same-shape stand-in sheet for offline demos — output from a synthetic
sheet is labelled as such and is not a finding about real products.

## Tools: enriching non-visual tags

Some tags are not visual. No photo reveals whether a model won an `award`, appeared in a
`benchmark`, or carries an `energy rating` — that knowledge is outside the image. `--enrich` adds an
`enrich` node that looks the product up:

```
load_image → recall ─(hit)──────────────────────────────────────────► END
                    └(miss)─► tag_image → enrich → validate_tags → remember → END
```

```bash
python -m content_analysis_agent.cli tag --input data/test --enrich --search-tool mock
python -m content_analysis_agent.cli tag --input data/test --enrich --search-tool anthropic
```

Two backends sit behind one protocol, mirroring `vlm.py`: `mock` is deterministic and offline;
`anthropic` uses Claude's **server-side** `web_search` tool, so the search runs on Anthropic's
infrastructure with no second API key and no client-side tool loop.

Three properties make this safe to add:

- **A tool can only suggest.** Enrichment appends candidates to `raw_tags`; `validate_tags` still
  decides. Nothing outside the controlled vocabulary can reach the output, whatever a search returns.
- **A search outage is non-fatal.** Enrichment is additive, so a failed lookup logs and returns the
  tags the model already produced rather than losing them.
- **Enriched results are cached separately.** The memory key includes whether enrichment ran, so a
  plain run's cache is never served for an enriched request.

## Agent workflow metrics

The scores in `eval` measure *tagging quality* against ground-truth labels, so they only run on the
8 labelled images. These measure how the agent *behaves*, need no labels, and therefore work on the
107 unlabelled test images — and in production, where labels never exist. Printed by both `tag` and
`eval`:

```
Agent workflow metrics (no labels required)
  Hallucination   : 0.000 (0/10 proposed tags out of vocabulary)
                    0.000 of answered images proposed at least one
  Latency (model) : p50 1100 ms | p95 1100 ms | 1 call(s)
  Latency (tool)  : p50 7943 ms | p95 7943 ms | 1 call(s)
  Efficiency      : cache hit 0.000 | 0 retr(ies) | 0 failure(s) (0.000)
```

Timing them separately is what shows enrichment costs ~7.5 s against ~1 s for the tagging call —
which is why `--enrich` is opt-in.

- **Hallucination** is the share of proposed tags falling outside the vocabulary. `validate_tags`
  drops them silently, so without this the failure is invisible. It needs no ground truth, which
  makes it the one quality signal available on live traffic — and the right thing to alert on.
- **Latency** is reported as p50/p95 rather than a mean, since tail latency is what a user notices.
  Model and tool calls are timed separately so enrichment's cost is attributable.
- **Efficiency** is what the run actually cost: cache hit rate, retries, and failures per image.

## Tag taxonomy

43 tags across 12 general categories, including `physical design` (front / side / back / multiple
angles, colour, case), `feature graphics` (camera, battery life, sound quality, gaming,
sustainability, and more), `usage scene` (indoor, outdoor, transport), plus standalone tags such as
`accessories`, `awards`, `dimension`, `energy rating`, and `whats in the box`.

Run `python -m content_analysis_agent.cli taxonomy` to print the full hierarchy. A few specifics that
appear in the training data but not in the original specification (`left`, `right`) are tracked
separately under `observed_extra` and merged in, so the model may predict them and evaluation does
not penalise them.

## Data and evaluation

```
data/
  train/<Category>/<Model>/['physical design', 'side angle', 'left'].jpg
  test/<Category>/<Model>/amazon_co_uk_<ASIN>_Main_Carousel_<n>_<date>.jpg
```

Ground truth is encoded directly in the **training** filenames as a literal Python list, so every
training image doubles as a labelled example and evaluation needs no separate annotation file. Test
images carry no labels; their category and model are inferred from the folder path.

Image files are gitignored — only the folder skeleton is committed, so a fresh clone starts with an
empty dataset.

`eval` reports micro and macro precision / recall / F1, sample-averaged Jaccard, exact-match rate,
and per-tag support. Metrics are computed by hand rather than via scikit-learn, to keep the
definitions explicit and the dependency list small.

### Measured results

Claude Haiku 4.5 on the 8 labelled images:

| Run | micro-F1 | Jaccard | exact-match |
| --- | --- | --- | --- |
| prior top-3 baseline | 0.638 | 0.469 | 0.000 |
| zero-shot | 0.630 | 0.517 | 0.125 |
| **`--few-shot 8`** | **0.875** | **0.812** | **0.625** |

Few-shot is what makes the difference: zero-shot does not beat a baseline that ignores the image,
while eight examples clear all three success criteria. Groq's `qwen3.8-27b` reaches 0.640 with
`--few-shot 2` (its 3-image-per-request ceiling). Eight images is a small sample — treat this as
directional, not a benchmark.

### What counts as success

Defined up front so a score can be judged rather than merely reported:

1. **Beat the model-free baselines.** An agent that cannot outscore "always guess the most common
   tags" is adding nothing.
2. **Micro-F1 ≥ 0.75.** Tags feed a human review queue, so pooled per-tag correctness matters more
   than getting whole sets exactly right.
3. **Exact-match ≥ 0.40** — roughly four in ten suggestions accepted untouched.

Every `eval` run prints the comparison automatically (`--no-baseline` to skip it):

```
                                                    micro-F1   Jaccard   exact
agent                                                  0.615     0.469   0.000
constant (physical design)                             0.516     0.354   0.000
prior top-3 (physical design, side angle, camera)      0.638     0.469   0.000

Beats best baseline : NO (0.615 vs 0.638)
```

That example is the **mock** provider, and it is instructive: the mock loses to a baseline that
ignores the image entirely. On this dataset the floor is unusually high because `physical design`
appears in all 8 labels, so a constant guess already scores 0.516. Both baselines are derived from
the ground-truth labels themselves, which makes them an *optimistic* floor — they start out knowing
the tag distribution the agent has to infer from pixels.

## Requirements

Python 3.11. Five direct dependencies: `langgraph`, `anthropic`, `openai`, `streamlit`, and
`python-dotenv`. See `requirements.txt`.
