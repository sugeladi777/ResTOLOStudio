from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from app.runtime import AppRuntime
from app.windows import ReSTOLOStudioWindow


def create_application(project_root: Path | None = None) -> tuple[QApplication, ReSTOLOStudioWindow]:
    root = Path(project_root or Path(__file__).resolve().parents[1])
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    runtime = AppRuntime.create(root)
    window = ReSTOLOStudioWindow(runtime)
    return app, window


def main() -> None:
    app, window = create_application()
    window.show()
    sys.exit(app.exec_())
