"""Raw-frame history definition, independent of the decoder prediction lag."""
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class HistoryConfig:
    # Includes current; the last element is always frame t.
    length: int = 3
    stride_frames: int = 1
    missing_policy: str = "repeat_earliest"

    def __post_init__(self):
        if self.length < 1 or self.stride_frames < 1:
            raise ValueError("history length and stride_frames must be positive")
        if self.missing_policy not in {"repeat_earliest", "drop"}:
            raise ValueError("history missing_policy must be repeat_earliest or drop")

    @classmethod
    def from_config(cls, config: Mapping):
        return cls(**config.get("history", {}))

    def frames(self, current: int) -> tuple[int, ...]:
        frames = tuple(current - offset * self.stride_frames for offset in reversed(range(self.length)))
        if frames[0] < 0 and self.missing_policy == "drop":
            raise ValueError("insufficient past history")
        return tuple(max(0, frame) for frame in frames)

    def as_dict(self):
        return asdict(self)
