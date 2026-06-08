# ReSTOLO Studio

Unified desktop application for:

- Nanonis STM control and scan capture
- ReSTOLO inference, annotation, and training
- Session-based result management

## Run

```powershell
pip install -r requirements.txt
python main.py
```

## Conda Environment

Recommended environment for this workstation:

- Python `3.10`
- PyTorch `2.5.1`
- CUDA `12.1`
- GPU verified on `NVIDIA GeForce RTX 4060`

Create and use the environment with:

```powershell
conda env create -f environment.yml
conda activate restolo-py310
python main.py
```

If the environment already exists, update it with:

```powershell
conda env update -f environment.yml --prune
```

## Structure

- `main.py`: slim launcher entrypoint
- `app/bootstrap.py`: application startup and Qt bootstrapping
- `app/runtime.py`: runtime/service assembly
- `app/windows/`: top-level application windows
- `app/windows/studio_window.py`: main studio window entry
- `app/windows/studio_panels.py`: acquisition and results panel assembly
- `app/windows/studio_actions.py`: window-level actions for Nanonis, sessions, and inference
- `app/windows/__init__.py`: exports the public studio window entrypoints
- `app/legacy/`: explicit boundary for historical base UI
- `app/legacy/workbench_impl.py`: thin legacy shell that now delegates layout and behavior to focused modules
- `app/legacy/workbench_ui.py`: extracted legacy UI theme and widget helpers shared by the base window
- `app/legacy/workbench_layout.py`: extracted legacy window construction and widget layout assembly
- `app/legacy/workbench_data.py`: extracted legacy data loading, SXM conversion, annotation, and model-loading helpers
- `app/legacy/workbench_training.py`: extracted legacy YOLO and ResNet training workflow helpers
- `app/legacy/workbench_runtime.py`: extracted legacy UI state switching and training/inference callback helpers
- `app/legacy/workbench_state.py`: extracted legacy tab-state and button-state logic
- `app/legacy/workbench_bindings.py`: centralizes legacy method binding so the base shell file stays smaller
- `app/services/__init__.py`, `app/ui/__init__.py`, `app/utils/__init__.py`: re-export the package-level public API instead of leaving empty shells
- `app/core/`: application-level paths and core primitives
- `app/core/paths.py`: single source of truth for bundled assets, sessions, config, and YOLO resource files
- `app/`: PyQt5 application and services
- `assets/`: bundled application resources separated from source code
- `assets/models/`: local model directory for detection/classification weights and `classes.yaml`
- `assets/config/error_patterns.yaml`: user-facing error matching rules
- `nanonis/`: Nanonis TCP backend
- `ml/`: YOLO / ResNet training and inference logic
- `ml/models/yolov5m_molecule.yaml`: retained custom YOLO architecture used by the desktop training flow
- `sessions/`: scan, inference, and training outputs

## Repository Hygiene

- runtime cache directories such as `__pycache__/` are not kept in the repository
- IDE metadata such as `.vs/`, `.suo`, and workspace index files are not kept in the repository
- training artifacts are not kept under source directories
- upstream deployment helpers that are not part of the desktop app workflow are removed instead of being carried in-tree
- optional upstream experiment-tracking integrations are reduced to local no-op compatibility when the desktop app does not use them
- `sessions/` is treated as runtime output and should be recreated locally instead of versioned
