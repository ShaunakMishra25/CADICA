# Scripts

Exploratory and diagnostic scripts. These are intentionally separate from the reusable
library code in `src/cadica/`.

| Script                   | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `check_dataset.py`       | Validate raw dataset layout and metadata     |
| `inspect_videos.py`      | Count videos per patient                     |
| `inspect_frames.py`      | Frame-count distribution per video           |
| `inspect_images.py`      | Image dimensions and formats                 |
| `inspect_model.py`       | Architecture and trainable parameter count   |
| `view_sequence.py`       | Plot consecutive-frame differences           |

Run them after installing the package (`pip install -e ".[dev]"`):

```bash
python scripts/check_dataset.py
python scripts/view_sequence.py --patient p1 --video v1
```

`view_sequence.py` uses a headless-safe Matplotlib backend and writes its figure to
`results/`.
