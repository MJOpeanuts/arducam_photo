import json
import numpy as np
import pytest

from arducam_photo import engine


class FakeCap:
    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self._opened = opened
        self.released = 0
        self.props = {}
        self.read_calls = 0
        self.set_calls = []

    def isOpened(self):
        return self._opened

    def set(self, prop, val):
        self.set_calls.append((prop, val))
        self.props[prop] = val
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        self.read_calls += 1
        item = self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]
        return item

    def release(self):
        self.released += 1


@pytest.fixture
def small_raw(monkeypatch):
    """Shrink the 108 MP geometry so tests are fast; logic is size-independent."""
    monkeypatch.setattr(engine, "RAW_ROWS", 8)
    monkeypatch.setattr(engine, "RAW_COLS", 12)
    monkeypatch.setattr(engine, "RAW_BYTES", 96)
    return 8, 12


def opener_for(cap):
    calls = []

    def opener(index, api):
        calls.append((index, api))
        return cap

    opener.calls = calls
    return opener


@pytest.fixture
def ccm_file(tmp_path):
    p = tmp_path / "arducam_108mp.json"
    ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    p.write_text(json.dumps({"ccms": [{"ct": 2800, "ccm": ident}, {"ct": 6500, "ccm": ident}]}))
    return str(p)


def good_raw(n=96):
    return (True, np.full((n,), 100, np.uint8))
