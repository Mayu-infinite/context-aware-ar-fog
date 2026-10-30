import importlib.util
import re
from pathlib import Path

from cafog.spatial import EngineConfig, make_profile

ROOT = Path(__file__).resolve().parents[1]
CSHARP_CONFIG = ROOT / "unity/CafogQuest/Assets/Cafog/Runtime/Core/CafogConfig.cs"


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_unity_config", ROOT / "scripts/export_unity_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csharp_fields() -> dict[str, set[str]]:
    """Public fields of each [Serializable] class in CafogConfig.cs."""
    text = CSHARP_CONFIG.read_text(encoding="utf-8")
    out: dict[str, set[str]] = {}
    for match in re.finditer(r"class (\w+)\s*\{(.*?)\n    \}", text, re.S):
        out[match.group(1)] = set(re.findall(r"public (?:\w+) (\w+) =", match.group(2)))
    return out


def test_every_python_setting_exists_in_csharp():
    data = load_exporter().unity_config(EngineConfig(profile=make_profile("healthy")))
    fields = csharp_fields()
    sections = {
        "map": "MapConfig",
        "value": "ValueConfig",
        "profile": "PatientProfile",
        "cue": "CueConfig",
    }
    for key, cls in sections.items():
        missing = set(data[key]) - fields[cls]
        assert not missing, f"{cls} lacks {sorted(missing)}"
    top = {k for k, v in data.items() if not isinstance(v, dict)}
    assert top <= fields["EngineConfig"]
    assert data["profile"]["deltaLim"] == 0.0


def test_yaml_profile_exports(tmp_path):
    config = EngineConfig.from_yaml(ROOT / "configs/spatial_engine.yaml")
    assert config.profile.name == "pd_fog"
    data = load_exporter().unity_config(config)
    assert data["cue"]["triggerDistanceM"] == config.cue.trigger_distance_m
