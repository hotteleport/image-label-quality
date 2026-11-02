"""Find exact and near duplicates, label conflicts and small images."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NEAR_DUPLICATE_DISTANCE = 4
MIN_SIDE = 200


@dataclass(frozen=True)
class Item:
    path: str
    label: str
    md5: str
    phash: str
    width: int
    height: int


def scan(root: Path) -> list:
    items = []
    for class_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                items.append(Item(
                    str(path.relative_to(root)), class_dir.name, digest,
                    str(imagehash.phash(image.convert("RGB"))), image.width, image.height,
                ))
    return items


class _Groups:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def duplicate_groups(items: list, max_distance: int = NEAR_DUPLICATE_DISTANCE) -> list:
    """Group files that are byte-identical or perceptually almost identical."""
    groups = _Groups(len(items))
    hashes = [imagehash.hex_to_hash(item.phash) for item in items]
    first_by_md5 = {}
    for i, item in enumerate(items):
        if item.md5 in first_by_md5:
            groups.union(i, first_by_md5[item.md5])
        first_by_md5.setdefault(item.md5, i)
        for j in range(i):
            if hashes[i] - hashes[j] <= max_distance:
                groups.union(i, j)

    members = defaultdict(list)
    for i in range(len(items)):
        members[groups.find(i)].append(i)
    return [sorted(group) for group in members.values()]


def summarise(items: list, groups: list) -> dict:
    repeated = [group for group in groups if len(group) > 1]
    conflicts = [g for g in repeated if len({items[i].label for i in g}) > 1]
    by_label = defaultdict(int)
    for item in items:
        by_label[item.label] += 1
    exact_extra = len(items) - len({item.md5 for item in items})
    return {
        "images": len(items),
        "by_label": dict(by_label),
        "exact_duplicate_files": exact_extra,
        "duplicate_groups": len(repeated),
        "redundant_files": sum(len(g) - 1 for g in repeated),
        "groups_with_label_conflict": len(conflicts),
        "small_images": sum(min(i.width, i.height) < MIN_SIDE for i in items),
    }
