import numpy as np
from PIL import Image

from label_quality.audit import duplicate_groups, scan, summarise
from label_quality.model import group_split
from run_experiment import propagate


def pattern(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).resize((256, 256))


def build(tmp_path):
    (tmp_path / "moose").mkdir()
    (tmp_path / "reindeer").mkdir()
    pattern(1).save(tmp_path / "moose/a.png")
    pattern(1).save(tmp_path / "moose/a_copy.png")
    pattern(2).save(tmp_path / "moose/b.png")
    pattern(3).save(tmp_path / "reindeer/c.png")
    pattern(3).save(tmp_path / "moose/c_conflict.png")
    pattern(4).resize((100, 100)).save(tmp_path / "reindeer/small.png")
    return scan(tmp_path)


def test_summary_counts_duplicates_conflicts_and_small_images(tmp_path):
    items = build(tmp_path)
    summary = summarise(items, duplicate_groups(items))
    assert summary["images"] == 6
    assert summary["exact_duplicate_files"] == 2
    assert summary["groups_with_label_conflict"] == 1
    assert summary["small_images"] == 1


def test_different_images_are_not_grouped(tmp_path):
    items = build(tmp_path)
    assert len(duplicate_groups(items)) == 4


def test_decision_reaches_the_whole_duplicate_group(tmp_path):
    items = build(tmp_path)
    groups = duplicate_groups(items)
    decisions = propagate(items, groups, {"moose/a.png": "not_photo"})
    named = {items[i].path for i in decisions}
    assert named == {"moose/a.png", "moose/a_copy.png"}


def test_group_split_never_separates_duplicates():
    y = np.array([0, 1] * 30)
    groups = np.repeat(np.arange(20), 3)
    train, test = group_split(y, groups, seed=1)
    assert not set(groups[train]) & set(groups[test])
