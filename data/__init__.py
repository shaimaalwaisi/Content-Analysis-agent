"""The dataset, and the sheet that describes it.

`meta_data.xlsx` ships alongside the train and test images, so the code that
reads it lives here too rather than in a package of its own: the join is on
image file name, and this is where the image file names are.

The brief opens by asking which content is popular and engaging. That is a
question about the *data* -- tags joined to image views -- not about how well
the agent tags, which is what `evaluation` measures. A tag can be predicted
perfectly and still be commercially dull, and the reverse.

Depends inward on `agent` for logging and label parsing; the agent imports
nothing from here, and the CLI wires the two together.
"""
from .metadata import (format_engagement, join_tags, load_metadata,
                       rows_from_sheet, tag_engagement,
                       write_synthetic_metadata)

__all__ = ["format_engagement", "join_tags", "load_metadata",
           "rows_from_sheet", "tag_engagement", "write_synthetic_metadata"]
