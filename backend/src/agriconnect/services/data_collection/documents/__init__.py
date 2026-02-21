"""
Document Collection Module

Scrapers spécialisés pour la collecte de documents techniques et bulletins.

Modules:
- fews: FEWS NET (Famine Early Warning System) documents  
- fews_pdf_harvester: PDF harvester pour FEWS documents
- sonagess: SONAGESS (Société Nationale de Gestion du Stock de Sécurité) bulletins

Usage:
    from backend.services.data_collection.documents import FewsPdfHarvester
    
    harvester = FewsPdfHarvester(output_dir="./output")
    harvester.scrape_all()
"""

from .fews_pdf_harvester import AnamBulletinScraper

# Backwards-compatible alias: some modules expect FewsPdfHarvester
FewsPdfHarvester = AnamBulletinScraper

__version__ = "1.0.0"

__all__ = [
    "fews",
    "fews_pdf_harvester", 
    "sonagess",
    "FewsPdfHarvester"
]
