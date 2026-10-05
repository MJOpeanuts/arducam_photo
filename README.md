# arducam_photo

Lightweight, importable one-shot photo capture for the Arducam B0494C (108 MP UVC), Windows 11 x64.
No GUI, database, server, preview, persistent profiles or installer.

## API

```python
from arducam_photo import CaptureConfig, capture, save_png

cfg = CaptureConfig(camera_index=0, api="msmf", path="native_108mp",
                    focus=187, ccm_path=r"C:\arducam\arducam_108mp.json")
result = capture(cfg)            # blocking: run it outside your UI thread
result.image                     # BGR uint8, 9000 x 12000 x 3 (width 12000, height 9000)
result.info                      # frozen config copy, settings, read counts, ccm_applied, duration
save_png(result.image, r"C:\photos\a.png")   # optional helper
```

- `CaptureConfig`: `camera_index`, `api` (`msmf` reference), `path` (`native_108mp` | `color_720p`, always explicit, no fallback), `focus`, `ccm_path`, `apply_ccm`, `stabilization_reads` (default 5, manufacturer starting point, not universal), `max_failed_reads`, `max_invalid_buffers`.
- `capture(config, cancel_event=None)` → `CaptureResult(image, width, height, info)`. Errors, all `CaptureError` subclasses: `CaptureConfigError`, `CaptureBusyError`, `CaptureCancelled`, `CameraOpenError`, `CameraSetupError`, `CameraReadError`, `RawBufferError`, `IspError`, `SaveError`.
- `save_png(image, path, overwrite=False)`: temp file in the same directory, decode-back verification, atomic publish, never replaces an existing file unless `overwrite=True`.
- Import has no side effects. One capture at a time per process (`CaptureBusyError`). The camera is opened and released inside each call.

## Paths

**native_108mp** (USB 3 required): requests 6000×9000 from the driver with `CAP_PROP_CONVERT_RGB=0`, applies focus, reads a bounded number of frames, validates each buffer (uint8, 108 000 000 elements/bytes) *before* reshaping to 9000×12000 Bayer, then black level −16 (saturating), `COLOR_BayerGR2RGB` demosaic (output is used as BGR, as in the reference), and CCM at 4000 K. 6000×9000 is the RAW transport size; the 12000×9000 result adds no pixels. No upscale, no substitution by 720p.
**color_720p**: 1280×720 at 10 fps, validated as uint8 (720,1280,3) BGR. No RAW processing, no CCM file. Works on USB 2 or 3 if the mode exists. USB speed is **not measured**; a missing mode/wrong buffer size only suggests a link limitation.

`info.settings` gives requested value, `cap.set` return value and driver read-back per setting; a read-back is not proof of physical effect. Autofocus/auto-exposure are not controlled unless you pass `focus`; automatic behaviour may differ between photos. A dark scene is never treated as a failure.

Blocking native calls (no interruptible timeout): `VideoCapture` open, `set`, `read`. Cancellation (`cancel_event`) is only checked between them; stabilization is bounded by read counts. COM: handled by OpenCV MSMF in the calling thread; the module does not call COM.

## CCM tuning file (`arducam_108mp.json`)

- In the Codespace it was **not found** by a filesystem search of the sandbox, and nothing is tracked by Git (`git ls-files` shows no JSON). It is in `.gitignore`. A Windows clone will not have it.
- Get it from the manufacturer repo (branch `Arducam108MPDemo`, commit `b6c599c80648041a0413df751d96f3eeb5f03270`) or copy your existing one to e.g. `C:\arducam\arducam_108mp.json` and pass it via `ccm_path` / `--ccm`. It is not bundled: redistribution rights are unverified.
- Validated: JSON object with non-empty `ccms`, each entry `ct` (positive, strictly increasing) and `ccm` (9 finite numbers). Missing/invalid file → `CaptureConfigError` before the camera is opened. `info.ccm_applied` records whether correction ran.

## References and licence

Reference demo: https://github.com/ArduCAM/ArducamUVCPythonDemo (branch `Arducam108MPDemo`, commit `b6c599c…`; files `arducam_demo.py`, `camera.py`, `isp.py`, `utils.py`) and the Windows quick-start guide at docs.arducam.com. The licence could not be verified from the sandbox (network blocked), so **no code or resources were copied**; the sequence (CONVERT_RGB=0, 5 stabilising reads, reshape 9000×12000, −16, BayerGR2RGB, CCM at 4000 K with the reference interpolation/rot180/transpose) was re-implemented. The CCM is applied in float32 by 256-row chunks (the reference uses float64 over the whole image); expect tiny rounding differences. Do not mix OpenCV distributions providing `cv2`.

## Windows PowerShell hardware test

```powershell
git clone https://github.com/MJOpeanuts/arducam_photo; cd arducam_photo
git checkout copilot/init-arducam-photo   # or the PR branch
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
python -c "import cv2,numpy;print(cv2.__version__,numpy.__version__)"   # expect 5.0.0 / 2.5.3 (opencv-python 5.0.0.93)
pip list | findstr /i opencv                                             # exactly one distribution
$ccm = "C:\arducam\arducam_108mp.json"; Test-Path $ccm
python examples\capture_cli.py --path native_108mp --ccm $ccm --focus 187 -o $env:TEMP\n1.png -v
python -c "import cv2;i=cv2.imread(r'$env:TEMP\n1.png');print(i.shape)"  # (9000, 12000, 3)
ii $env:TEMP\n1.png
1..3 | % { python examples\capture_cli.py --path native_108mp --ccm $ccm -o $env:TEMP\s$_.png }
python examples\capture_cli.py --path color_720p -o $env:TEMP\p720.png
python examples\use_as_library.py $ccm $env:TEMP\lib.png
```
Close any camera app first; use `--overwrite` to replace existing files.

## Status

| | |
|---|---|
| Implemented | both paths, CCM loader/ISP, PNG helper, CLI + library example, Windows CI workflow |
| Tested automatically (simulated camera, tiny Bayer images) | buffer validation, transient frames, bounds, open/read failure, no fallback, channel order, release on success/error, concurrency, import side effects, PNG round-trip, CCM errors, 720p isolation |
| Measured (Linux sandbox, synthetic) | ISP at 9000×12000 incl. CCM: ~0.6 s, ~0.76 GB peak RSS; not representative of Windows |
| Tested on hardware | **nothing yet** (no camera in Codespace) |
| To verify on hardware | direct native capture without preview, 12000×9000 content, focus/colour vs. demo, successive captures, camera freed, 720p, import from a second program |

The milestone is **not** validated by simulated tests alone. Later (not here): separate Preview/Settings and Capture modes; a capture never restarts the preview.
