# ReSTOLO Studio

ReSTOLO Studio is a desktop application for STM-oriented workflows:

- acquire scan data from Nanonis / STM control paths
- inspect and annotate images, including `.sxm` conversion
- train YOLO and ResNet models
- run inference and manage experiment sessions

## Current architecture

- `main.py`: application entrypoint
- `app/bootstrap.py`: creates `QApplication`, runtime, and main window
- `app/core/`: shared primitives such as project paths and session/result models
- `app/runtime.py`: shared runtime services and project paths
- `app/windows/`: studio window, acquisition panel, results panel, studio controller, training controller, runtime controller, and studio shell
- `app/windows/training_controller.py`: studio-owned YOLO / ResNet training workflows
- `app/windows/runtime_controller.py`: studio-owned runtime callbacks and UI state updates
- `app/services/`: application services such as config, annotation state/file IO, acquisition workflows, image loading workflows, inference, resource loading, session/result workflows, training workflows, training runners, SXM conversion, dataset preparation, and training job persistence
- `app/ui/`: reusable Qt widgets such as the annotation tool and loss dialog; widgets should focus on interaction/presentation rather than file IO
- `app/utils/`: model, training, inference, error matching, and SXM parsing helpers
- `nanonis/`: Nanonis TCP and scan workflow integration
- `ml/`: YOLO / ResNet training and inference implementation
- `assets/`: default models, classes, and app config assets
- `sessions/`: generated scan, training, and inference outputs

## Architecture notes

The project now uses a clearer modular architecture:

- the studio window remains the Qt shell
- runtime dependencies are created in `app/runtime.py`
- annotation state now has explicit core models and an `annotation_service`
- annotation file loading/saving has been moved out of `AnnotationTool`
- Studio data/model loading flows now live in `studio_controller`
- Studio training flows now live in `training_controller`
- Studio runtime/state flows now live in `runtime_controller`
- image loading, SXM conversion coordination, annotation state sync, and inference image preparation now live in `image_workflow_service`
- default model paths, classes files, and dataset path loading now live in `resource_loader_service`
- Nanonis connection configuration, scan geometry assembly, and scan result persistence now live in `acquisition_workflow_service`
- session creation, selection, and result persistence orchestration now lives in `session_workflow_service`
- training session binding, dataset preparation, and training plan generation now live in `training_workflow_service`
- YOLO / ResNet execution wrappers and ResNet log parsing now live in `training_runner_service`
- reusable SXM conversion and dataset preparation logic now lives under `app/services/`
- sessions and result records now use explicit core models instead of raw dictionaries
- training runs now have a dedicated persistence path via `training_job_service`

## Recommended next steps

- continue shrinking Qt-window-level orchestration where it still leaks workflow details
- extend the smoke coverage from workflow integration into more realistic file- and UI-driven scenarios

## Verification

- `python -m compileall app tests`
- `pytest tests/test_resource_loader_service.py tests/test_file_pipeline_smoke.py tests/test_workflow_integration_smoke.py tests/test_inference_workflow_service.py tests/test_acquisition_workflow_service.py tests/test_training_runner_service.py tests/test_training_workflow_service.py tests/test_image_workflow_service.py tests/test_session_workflow_service.py tests/test_studio_controllers.py tests/test_runtime_smoke.py tests/test_result_store_service.py`
