"""Second program importing the same function (no copied code).

python examples/use_as_library.py C:\\cfg\\arducam_108mp.json out.png
"""

import sys

from arducam_photo import CaptureConfig, capture_and_save

ccm, out = sys.argv[1], sys.argv[2]
result = capture_and_save(CaptureConfig(path="native_108mp", ccm_path=ccm), out)
print(result.width, result.height, result.info.ccm_applied, result.info.settings)
