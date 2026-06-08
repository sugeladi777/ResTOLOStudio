from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnnotationBox:
    cls: int
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_tuple(cls, payload: tuple[int, float, float, float, float]) -> "AnnotationBox":
        return cls(
            cls=int(payload[0]),
            x=float(payload[1]),
            y=float(payload[2]),
            w=float(payload[3]),
            h=float(payload[4]),
        )

    def to_tuple(self) -> tuple[int, float, float, float, float]:
        return (self.cls, self.x, self.y, self.w, self.h)


@dataclass
class AnnotationState:
    images: list[str] = field(default_factory=list)
    annotations: dict[str, list[AnnotationBox]] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=lambda: [str(i) for i in range(10)])
    current_index: int = 0

    def normalized(self) -> "AnnotationState":
        images = list(self.images)
        annotations = {path: list(self.annotations.get(path, [])) for path in images}
        for path, boxes in self.annotations.items():
            if path not in annotations:
                annotations[path] = list(boxes)

        if not images:
            current_index = 0
        else:
            current_index = max(0, min(self.current_index, len(images) - 1))

        class_names = list(self.class_names) if self.class_names else [str(i) for i in range(10)]
        return AnnotationState(
            images=images,
            annotations=annotations,
            class_names=class_names,
            current_index=current_index,
        )
