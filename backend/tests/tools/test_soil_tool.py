import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agriconnect.tools import soil as soil_mod


def test_get_diagnosis_from_soilgrids_basic():
    tool = soil_mod.SoilDoctorTool()
    sample = {
        "layers": {
            "sand": {"0-5cm": {"mean": 593}},
            "clay": {"0-5cm": {"mean": 180}},
            "phh2o": {"0-5cm": {"mean": 66}},
            "soc": {"0-5cm": {"mean": 82}},
            "nitrogen": {"0-5cm": {"mean": 74}},
            "cec": {"0-5cm": {"mean": 92}}
        }
    }

    res = tool.get_diagnosis_from_soilgrids(sample, observation="sec")
    assert "identite_pedologique" in res
    assert res["identite_pedologique"]["nom_local"] != ""
    assert "gestion_eau" in res
    assert "besoin_irrigation_estime" in res["gestion_eau"]


def test_analyze_ph_boundaries():
    tool = soil_mod.SoilDoctorTool()
    assert "Acide" in tool._analyze_ph(5.0)
    assert "Alcalin" in tool._analyze_ph(8.5)
    assert "Équilibré" in tool._analyze_ph(6.5)
