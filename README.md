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
| `evaluate.py` | Multi-label metrics computed with plain set arithmetic. |
| `memory.py` | Persistent tag memory (SQLite), so identical requests skip the model. |
| `fewshot.py` | Turns the labelled training images into few-shot demonstrations. |

Because every entry point goes through `build_graph`, the graph is the extension point: inserting a
node between `tag_image` and `validate_tags` — say, an enrichment step that looks up non-visual tags
such as `awards` or `benchmark` — reaches the CLI, the pipeline, evaluation, and the UI at once.

### Providers

| `--provider` | Model | Notes |
| --- | --- | --- |
| `anthropic` | `claude-sonnet-5` | Default. Needs `ANTHROPIC_API_KEY`. |
| `openai` | `gpt-4o` | Needs `OPENAI_API_KEY`. |
| `mock` | — | No network, no key. Returns a fixed guess per category; useful for wiring and demos, **not** for measuring accuracy. |

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
