# arducam_photo

Lightweight, importable one-shot photo capture for the Arducam B0494C (108 MP UVC), Windows 11 x64.
The repository has three layers: the **library** (`arducam_photo`), the **CLI examples** (`examples/`) and the **graphical app** (`arducam_photo.app`, PySide6). No Streamlit, browser, server, SQLite or installer.

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

- The existing tracked tuning file is bundled at `src/arducam_photo/resources/arducam_108mp.json`, with its contents unchanged. `.gitignore` allows this specific resource while excluding other files with the same name.
- For the library and CLI, pass this file or your own tuning file via `ccm_path` / `--ccm`; the capture engine still requires an explicit path.

**Default in the GUI**: the app uses the package resource `src/arducam_photo/resources/arducam_108mp.json` (loaded via `importlib.resources`, independent of the current directory; included in wheels through `pyproject.toml`). No file selection is required at first launch. Choosing another file ("Avancé : utiliser un autre arducam_108mp.json…") is an optional override and takes priority over the resource. If the resource is missing or invalid, the 108 MP capture is refused with a precise error; the correction is never silently disabled. Typical flow: `git pull`, `pip install .[gui]`, `arducam-capture`.
- Validated: JSON object with non-empty `ccms`, each entry `ct` (positive, strictly increasing) and `ccm` (9 finite numbers). Missing/invalid file → `CaptureConfigError` before the camera is opened. `info.ccm_applied` records whether correction ran.

## References and licence

Reference demo: https://github.com/ArduCAM/ArducamUVCPythonDemo (branch `Arducam108MPDemo`, commit `b6c599c…`; files `arducam_demo.py`, `camera.py`, `isp.py`, `utils.py`) and the Windows quick-start guide at docs.arducam.com. The licence could not be verified from the sandbox (network blocked); no manufacturer code was copied. The tuning resource is the existing repository file, relocated without changes. The sequence (CONVERT_RGB=0, 5 stabilising reads, reshape 9000×12000, −16, BayerGR2RGB, CCM at 4000 K with the reference interpolation/rot180/transpose) was re-implemented. The CCM is applied in float32 by 256-row chunks (the reference uses float64 over the whole image); expect tiny rounding differences. Do not mix OpenCV distributions providing `cv2`.

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

The milestone is **not** validated by simulated tests alone.

## Graphical application (PySide6)

Install and run (Windows, no PowerShell activation required for the `.cmd`):

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[gui]"
.venv\Scripts\python.exe -m arducam_photo.app        # single documented command
run_arducam_capture.cmd                                 # double-click launcher
```

Quick trial on your PC: close other camera apps, start the app, choose *Photo 108 MP*, pick the camera (use "Détecter" if needed; the index is not a stable identity), leave the bundled CCM selected by default, open **Preview / Réglages**, set the focus (0–1023, numeric + slider; no autofocus or calibrated distance), **Enregistrer**, go to **Capture**, press **PHOTO**. PNGs go to `Pictures\ArducamCapture` (real, possibly OneDrive-redirected folder; changeable). Unique file names; nothing is overwritten, moved or deleted.

**Modes.** Start: nothing is opened. Preview: 1280×720 at 30 fps (108 MP path, "if available") or 10 fps (720p), acquired and displayed FPS measured separately, manual focus with requested vs read-back value, unsaved-changes indicator and prompt (save / discard / stay). Capture: no video and no camera read while waiting; the saved profile is frozen at trigger time, the existing engine runs on the worker thread, the focus is reapplied each capture, the camera is released after (also on error), and the app stays in Capture (the preview never restarts by itself). A second trigger and mode changes are refused during a capture. Progress is shown by steps, without percentage.

**Architecture.** Qt-free: `app/profile.py` (profile + injectable `ProfileStore`, JSON in `%LOCALAPPDATA%\ArducamCapture`), `app/paths.py`, `app/preview.py` (preview adapter), `app/controller.py` (single worker owning every camera operation; one latest frame, one outstanding notification), `app/model.py` (mode orchestration). Qt: `app/widget.py` (reusable `CameraWidget`, host passes its own `SessionModel`/stores) and `app/main.py`. Frames are deep-copied into `QImage` so Qt never references freed numpy buffers. The profile holds camera id (index only: flagged fragile), path, API, requested focus and read-back; not the RAW transport nor the preview mode. A missing/invalid CCM file refuses the native capture; the correction is never silently disabled. Corrupt JSON files are kept (renamed `.corrupt-*`), never deleted.

**Limits.** OpenCV open/set/read/release are native and non-interruptible: no effective timeout or cancellation is promised; closing the window waits (up to 10 s) for the worker and cannot abort a blocked call. The 14.2 s of the first test is neither a guarantee nor an end-to-end UI measure. The "last photo" shown is from the current session only. USB speed is never measured. Do not mix OpenCV distributions.

**Tests run (Linux sandbox, simulated camera, `python -m pytest`, offscreen Qt):** engine/storage regressions, preview/capture transitions, no reads while waiting in Capture, no automatic preview, profile save/reload, unsaved-change decisions, saved focus passed to the engine, frozen profile, mutual exclusion, error/camera release, double trigger, clean shutdown, QImage buffer lifetime. The Windows CI workflow runs the same suite; no Windows run was done here.

**Hardware validation still to do:** 30-minute preview; visual focus tuning; profile persistence after relaunch; 20 consecutive captures; dimensions, sharpness, colours; UI responsiveness, memory, latency; disconnect/reconnect; close without leaving the camera busy. The app is **not** validated on hardware.

### Présentation Preview / Capture

- Interface anthracite, Segoe UI, image dominante sur fond sombre, sans étirement ni recadrage. Redimensionner ne change ni la résolution ni la cadence caméra. **Plein écran** agrandit seulement le flux ; **Échap** revient aux réglages.
- Sous le preview : focus manuel, **Enregistrer**, un seul statut de sauvegarde et **Capture** (gris clair, changement de mode uniquement). Sous la dernière photo : **PHOTO** (orange), deux boutons d’ouverture de style identique, **Preview** (gris clair).
- **Diagnostic**, fermé initialement, regroupe FPS, consigne/lecture de focus et limites de vérification, backend, chemins, correction couleur, dimensions, durées et détails des erreurs. Une erreur bloquante garde un résumé visible hors Diagnostic.
- Les SVG officiels `squirrel` et `refresh-cw` proviennent de [Lucide 0.468.0](https://github.com/lucide-icons/lucide/tree/0.468.0/icons). Ils sont embarqués dans le package avec leur [notice ISC](src/arducam_photo/resources/icons/LICENSE), sans accès réseau au lancement.

Tests d’interface (Linux Qt offscreen ; aucune caméra nécessaire) :

```sh
PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m pytest
PYTHONPATH=src QT_QPA_PLATFORM=offscreen QT_SCALE_FACTOR=1.25 python -m pytest tests/test_widget.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen QT_SCALE_FACTOR=1.5 python -m pytest tests/test_widget.py
python -m pip wheel . --no-deps -w /tmp/arducam-wheel
```

Exécuté pour cette refonte : **76 tests réussis** pour la suite complète à 100 %, **23 tests d’interface réussis** à chacun des facteurs 125 % et 150 %, construction du wheel et chargement des SVG depuis un package installé hors du dépôt.

Ces tests couvrent l’unicité du statut (y compris un échec de sauvegarde), le focus restauré, le redimensionnement sans nouvelle frame, les proportions, le plein écran/Échap, le Diagnostic, les ressources SVG, les états normal/survol/focus/désactivé, les boutons d’ouverture et les transitions du contrôleur. Les commandes restent accessibles dans les zones logiques 1280 × 720, 1024 × 576, 854 × 480 et 838 × 400 (marge pour la barre des tâches et les décorations), sans agrandissement forcé de la fenêtre, même Diagnostic ouvert.

Les facteurs Qt 100/125/150 % sous Linux ne remplacent **pas** une validation Windows : vérifier encore Segoe UI, survol/focus/désactivation, plein écran et absence de texte coupé sur un écran Windows 1280 × 720 à chaque mise à l’échelle, ainsi que tous les essais matériels ci-dessus.

Captures du **rendu réel Qt**, à 1280 × 720, issues du test de transitions avec caméra simulée (frames noires), et non de maquettes. L’heure affichée vient du fichier produit par le test ; aucune image de démonstration n’est chargée par l’application.

| Preview / Réglages | Capture après enregistrement |
| --- | --- |
| ![Preview rendu par Qt](docs/screenshots/preview.png) | ![Capture rendu par Qt](docs/screenshots/capture.png) |
