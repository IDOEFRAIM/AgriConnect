import os
import sys
import datetime

# --- IMPORT CONFIGURATION & MODULES ---
try:
    from ... import config
    from .fanfar_flood_risk import FanfarUltimateScraper
except ImportError:
    # Fallback pour exécution directe
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import config
        from fanfar_risk import FanfarUltimateScraper
    except ImportError:
        print("ERREUR CRITIQUE : Impossible de charger 'config.py' ou 'fanfar_risk.py'.")
        sys.exit(1)

# --- CONFIGURATION LOCALE ---
# Vous pouvez déplacer cette liste dans config.py si vous voulez la partager
VILLES_A_SURVEILLER = [
    "Niamey",
    "Bamako",
    "Ouagadougou",
    "Lagos",
    "Cotonou"
]

def generer_rapport():
    """
    Orchestre le scan des risques d'inondation et génère un rapport Markdown.
    """
    # Définition des chemins via la config
    base_dir = getattr(config, 'BASE_DIR', os.getcwd())
    rapports_dir = os.path.join(base_dir, "rapports")
    
    if not os.path.exists(rapports_dir):
        os.makedirs(rapports_dir, exist_ok=True)

    print("=== DÉMARRAGE DU GÉNÉRATEUR DE RAPPORT FANFAR ===")
    print(f"Villes cibles : {', '.join(VILLES_A_SURVEILLER)}")
    
    # Initialisation du scraper
    # Il récupère automatiquement les options (Headless, User-Agent) depuis config.py
    scraper = FanfarUltimateScraper()
    
    date_jour = datetime.datetime.now().strftime("%Y-%m-%d")
    heure_scan = datetime.datetime.now().strftime("%H:%M")
    
    resultats = []

    # 1. SCAN DES VILLES
    for ville in VILLES_A_SURVEILLER:
        print(f"\n[SCAN] Analyse de {ville} en cours...")
        try:
            res = scraper.get_risk(ville)
            resultats.append(res)
            
            # Feedback visuel simple
            icon = "✅" if res.risk_score == 0 else "⚠️"
            print(f"   -> {icon} Résultat : {res.risk_level} (Station: {res.station_name})")
            
        except Exception as e:
            print(f"   -> ❌ Erreur critique : {e}")

    # 2. GÉNÉRATION DU FICHIER MARKDOWN
    nom_fichier = f"Bilan_Inondations_{date_jour}.md"
    fichier_sortie = os.path.join(rapports_dir, nom_fichier)
    
    # URL source pour le rapport
    url_source = getattr(config, 'URL_FANFAR_PIV', "https://fanfar.eu/fr/piv/")

    with open(fichier_sortie, "w", encoding="utf-8") as f:
        # En-tête du rapport
        f.write(f"# 🌊 Rapport de Risque d'Inondation (FANFAR)\n")
        f.write(f"**Date :** {date_jour} à {heure_scan}\n")
        f.write(f"**Source :** [Portail FANFAR]({url_source})\n\n")
        
        f.write("---\n\n")
        
        # Résumé Rapide (Tableau)
        f.write("## 📊 Synthèse Rapide\n\n")
        f.write("| Ville | Niveau de Risque | Score | Station Détectée |\n")
        f.write("|-------|------------------|-------|------------------|\n")
        for res in resultats:
            # Code couleur
            icon = "🟢"
            if res.risk_score == 1: icon = "🟡"
            if res.risk_score == 2: icon = "🟠"
            if res.risk_score >= 3: icon = "🔴"
            if res.risk_level.startswith("Lieu Introuvable"): icon = "❓"
            
            f.write(f"| {res.location} | {icon} {res.risk_level} | {res.risk_score}/3 | {res.station_name} |\n")
        
        f.write("\n---\n\n")
        
        # Détails par ville
        f.write("## 📍 Détails par Localité\n\n")
        for res in resultats:
            f.write(f"### {res.location}\n")
            f.write(f"- **Station Hydrologique** : {res.station_name}\n")
            f.write(f"- **Niveau de Risque** : {res.risk_level}\n")
            f.write(f"- **Méthode de détection** : {res.source_type}\n")
            f.write(f"- **Détails techniques** : `{res.details}`\n")
            
            if res.screenshot_path:
                # Calcul du chemin relatif pour que l'image s'affiche bien dans le Markdown
                # Le markdown est dans rapports/, les images dans rapports/captures/
                # Donc le lien doit être captures/image.png
                nom_image = os.path.basename(res.screenshot_path)
                rel_path = f"captures/{nom_image}"
                
                f.write(f"\n**Preuve Visuelle :**\n")
                f.write(f"![Capture {res.location}]({rel_path})\n")
            else:
                f.write("\n*Aucune capture d'écran disponible.*\n")
            
            f.write("\n---\n")

    print(f"\n✅ Rapport généré avec succès : {fichier_sortie}")
    print("Les captures d'écran sont dans le sous-dossier 'captures/'.")

if __name__ == "__main__":
    generer_rapport()