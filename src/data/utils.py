import json
from pathlib import Path

here = Path(__file__).parent


def get_organ_labels():
    with open(str(here / 'semantic_organ_labels.json'), 'rb') as handle:
        labels = json.load(handle)
    return labels
