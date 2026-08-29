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

Useful flags: `--limit N` caps how many images are tagged, `--model` overrides the model id, and
`--provider` selects `anthropic` (default), `openai`, or `mock`.

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

Upload a single image, pick a provider in the sidebar, and press **Tag image**.

Uploaded files land in a temp path, so the agent cannot infer the product from the folder name the
way the CLI does. Fill in the **Product context** box (e.g. `Category: TV, Model: XR-65A95K`) to give
the model that hint. Choose the `mock` provider to try the interface with no API key.

## How it works

```
taxonomy.json  ->  prompts  ->  VLM client  ->  graph  ->  pipeline / evaluate  ->  CLI / UI
```

The agent itself is three nodes:

```
load_image  ->  tag_image  ->  validate_tags  ->  END
```

| Module | Responsibility |
| --- | --- |
| `taxonomy.py` / `taxonomy.json` | The controlled vocabulary as a two-level General → Specific hierarchy, editable without touching code. |
| `prompts.py` | Every LLM prompt, in one place, so wording can be tuned centrally. |
| `vlm.py` | Swappable providers behind one `VLMClient` protocol: Anthropic, OpenAI, and an offline mock. |
| `graph.py` | The LangGraph agent, including the validation step that drops out-of-vocabulary tags. |
| `pipeline.py` | Batch-tags a folder, tolerating per-image failures so one bad file cannot abort a run. |
| `evaluate.py` | Multi-label metrics computed with plain set arithmetic. |

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

## Requirements

Python 3.11. Five direct dependencies: `langgraph`, `anthropic`, `openai`, `streamlit`, and
`python-dotenv`. See `requirements.txt`.
