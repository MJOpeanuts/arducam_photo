"""CLI: python examples/capture_cli.py --path native_108mp --ccm C:\\cfg\\arducam_108mp.json -o photo.png"""

import argparse
import logging
import sys

from arducam_photo import CaptureConfig, CaptureError, capture_and_save
from arducam_photo.config import PATHS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", choices=PATHS, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--api", choices=["msmf", "dshow", "any"], default="msmf")
    p.add_argument("--focus", type=int)
    p.add_argument("--fps", type=float, help="Requested frame rate (driver readback is not a guarantee)")
    p.add_argument("--ccm", help="arducam_108mp.json (native_108mp)")
    p.add_argument("--no-ccm", action="store_true", help="native_108mp without color correction")
    p.add_argument("-o", "--output", required=True, help="PNG path")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    cfg = CaptureConfig(camera_index=a.index, api=a.api, path=a.path, focus=a.focus,
                        ccm_path=a.ccm, apply_ccm=not a.no_ccm, fps=a.fps)
    try:
        r = capture_and_save(cfg, a.output, overwrite=a.overwrite)
    except CaptureError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"saved {a.output}: {r.width}x{r.height} ccm_applied={r.info.ccm_applied} "
          f"duration={r.info.duration_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
