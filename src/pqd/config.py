"""Central settings. Everything else in the package reads its numbers from here."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "generated"
MODEL_DIR = PROJECT_ROOT / "models"

CLASSES = ["normal", "sag", "swell", "harmonics", "interruption", "transient", "noise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
DISTURBANCES = CLASSES[1:]  # every class except "normal"


@dataclass
class Config:
    f_nominal: float = 50.0
    v_rms_nominal: float = 230.0
    fs: int = 10_000
    window_ms: float = 200.0
    step_ms: float = 20.0
    seed: int = 42
    base_noise_std: float = 0.003   # per-unit
    min_cover: float = 0.2          # fraction of a window an event must cover to label it
    lead_s: float = 0.3
    min_gap_s: float = 0.25
    event_cycles: dict = field(
        default_factory=lambda: {
            "sag": [2, 30],
            "swell": [2, 30],
            "interruption": [2, 30],
            "harmonics": [5, 40],
            "noise": [5, 40],
        }
    )
    transient_ms: list = field(default_factory=lambda: [3.0, 10.0])

    def __post_init__(self) -> None:
        ratio = self.fs / self.f_nominal
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"fs ({self.fs}) must be a whole multiple of f_nominal ({self.f_nominal}) "
                "so that one cycle is a whole number of samples."
            )

    @property
    def cycle_samples(self) -> int:
        return int(round(self.fs / self.f_nominal))

    @property
    def window_samples(self) -> int:
        return int(round(self.fs * self.window_ms / 1000.0))

    @property
    def step_samples(self) -> int:
        return max(1, int(round(self.fs * self.step_ms / 1000.0)))

    @property
    def v_peak(self) -> float:
        return self.v_rms_nominal * 2 ** 0.5


def load_config(path: str | Path | None = None) -> Config:
    """Read config.yaml (if it exists) on top of the defaults."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return Config()
    try:
        import yaml
    except ImportError:
        return Config()
    data = yaml.safe_load(path.read_text()) or {}
    allowed = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in allowed})
