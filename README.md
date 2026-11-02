# Image Label Quality Audit

A moose vs reindeer classifier scored 82% on its own dataset. Was the model weak, or was the data? This project audits a web-scraped image dataset, reviews labels against a written policy, and tests whether cleaning actually helps.

**Built with:** Python, PyTorch (pretrained ResNet-18), scikit-learn, imagehash

## Key findings

| Finding | Result |
|---|---|
| Random sample usable as real, correctly labelled photos | **36%** (87 of 240 groups, 95% CI 30–43%) |
| Redundant duplicate files | **144 of 892** (101 exact, 43 near duplicates) |
| Duplicate groups filed under both classes | **2** |
| Accuracy inflated by duplicates crossing the split | **+4.4 points** (86.2% vs 81.8%) |
| Same model on verified real photos | **96.6%** (95% CI 92–100%, 87 images) |
| Effect of removing flagged training images | **No improvement** (95.4% vs 96.6%, one test image) |

**Conclusion:** the low score was mainly a data problem. The model was already good at the real task, and the test data was hiding that. Cleaning the training data did not improve it, so the most valuable step was building a trustworthy test set. Careful measurement avoided spending more effort on the wrong fix.

## Label policy

Each image gets one decision:

| Decision | Meaning | Examples found |
|---|---|---|
| `moose` / `reindeer` | Real photo of that animal | Wild and farmed animals |
| `not_photo` | Not a photograph of a live animal | Earrings, plush toys, socks, clip art, statues, paintings |
| `wrong_animal` | Real photo of another species | Elk, red deer, white-tailed deer, goat, horse, sea lion |
| `uncertain` | Cannot be decided from the image | A hoof only, animal barely visible |
| `relabel` | Real photo in the wrong class | Caribou filed as moose, one under a "Mr. Moose" caption |

## Method

1. **Audit.** Exact duplicates with MD5, near duplicates with perceptual hashing, and groups containing conflicting labels.
2. **Flag.** Pretrained ResNet-18 embeddings with logistic regression. Each image is scored by a model that never saw it or its duplicates.
3. **Review.** 159 flagged images, of which 84% had a real problem, against 58% in the random sample.
4. **Test set.** 240 random duplicate groups, chosen without using the model and reviewed by eye. The 87 verified real photos form the test set.
5. **Compare.** The same classifier trained on raw, deduplicated and reviewed data.

The test set is kept separate on purpose. Deleting images the model disagrees with from the test data would raise the score without improving the model.

## Limitations

- Review was AI-assisted, done with Claude from image thumbnails, and has one reviewer. There is no inter-annotator agreement, and some species calls may be wrong.
- Only flagged images in the training pool were reviewed, so many non-photos remain in the "reviewed" training set.
- The test set has 87 images, so small differences are not meaningful.
- Flagging ran on the full dataset before the test sample was drawn, so test images influenced flagging slightly.

## Running it

Images are not included because they were scraped from the web and may be under copyright. Provide a folder with `moose/` and `reindeer/` subfolders. The review files only match the original dataset.

```bash
pip install -r requirements.txt
python run_experiment.py /path/to/images
pytest
```

## Files

```text
label_quality/audit.py    duplicate detection and dataset summary
label_quality/model.py    embeddings, classifier and leakage-safe splits
run_experiment.py         full experiment
reviews/                  review decisions by file name
results/metrics.json      results quoted above
```

## How I used AI

This started from university coursework and was rebuilt with Claude Code. Claude wrote the code and carried out the visual review with me directing the project, so treat it as an AI-assisted study.
