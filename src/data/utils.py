import json
from pathlib import Path

from src import settings

here = Path(__file__).parent


IGNORE_CLASSES = [
    'gallbladder',
]


def get_organ_labels():
    with open(str(here / 'semantic_organ_labels.json'), 'rb') as handle:
        labels = json.load(handle)
    return labels


def get_label_mapping():
    mapping = settings.mapping
    organ_labels = settings.organ_labels
    content = {k: i for k, i in mapping.items() if i in organ_labels and i not in IGNORE_CLASSES}
    return content


def get_pa_label_mapping():
    return {0: 'vein', 1: 'artery'}
