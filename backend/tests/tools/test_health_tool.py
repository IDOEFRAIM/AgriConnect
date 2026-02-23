import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.src.agriconnect.tools import health as health_mod


def test_diagnose_unknown_crop():
    tool = health_mod.HealthDoctorTool()
    res = tool.diagnose("inconnue", "feuilles trouées")
    assert res["status"] == "Inconnu"


def test_diagnose_uncertain_and_identified():
    tool = health_mod.HealthDoctorTool()
    # Observations that don't match keywords
    res = tool.diagnose("maïs", "plante saine sans symptomes")
    assert res["status"] == "Incertain"

    # Observations that match keywords for Chenille Légionnaire
    res2 = tool.diagnose("maïs", "feuilles trouées et sciure visible", rate=20)
    assert res2["status"] == "Identifié"
    assert "Chenille" in res2["nom"] or "Striga" not in res2["nom"]
    assert res2["traitement_chimique"] != "Non requis (Seuil non atteint)"
