"""
Script de seed pour les zones agro-écologiques du Burkina Faso
Insère les 5 zones principales dans la base PostgreSQL

Anciennement: seed_zones.py (racine)
Nouvel emplacement: scripts/seed_zones.py
"""

import os
import sys
import uuid
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# URL de connexion depuis .env
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/agriconnect")

print("🔌 Connexion à PostgreSQL...")
engine = create_engine(DATABASE_URL)

# Données des 5 zones agro-écologiques
zones_data = [
    {
        "id": str(uuid.uuid4()),
        "name": "Boucle du Mouhoun - Dedougou",
        "region": "Boucle du Mouhoun",
        "coordinates": {"lat": 12.4667, "lng": -3.4833},
        "agro_type": "GRENIER",
        "description": "Grenier céréalier (maïs, sorgho, riz pluvial)"
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Hauts-Bassins - Bobo-Dioulasso",
        "region": "Hauts-Bassins",
        "coordinates": {"lat": 11.1767, "lng": -4.2967},
        "agro_type": "COTON",
        "description": "Zone cotonnière intensive"
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Sahel - Dori",
        "region": "Sahel",
        "coordinates": {"lat": 14.0333, "lng": -0.0333},
        "agro_type": "PASTORAL",
        "description": "Zone pastorale (élevage, cultures résilientes)"
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Centre-Ouest - Koudougou",
        "region": "Centre-Ouest",
        "coordinates": {"lat": 12.2528, "lng": -2.3650},
        "agro_type": "MARAICHAGE",
        "description": "Maraîchage et cultures vivrières"
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Sud-Ouest - Gaoua",
        "region": "Sud-Ouest",
        "coordinates": {"lat": 10.3333, "lng": -3.1833},
        "agro_type": "FRUITIER",
        "description": "Zone fruitière (mangue, anacarde)"
    }
]

print("\n🌍 Insertion des zones agro-écologiques...\n")

with engine.connect() as conn:
    for zone in zones_data:
        # Vérifier si la zone existe déjà
        check_query = text("SELECT COUNT(*) FROM zones WHERE name = :name")
        result = conn.execute(check_query, {"name": zone["name"]})
        count = result.scalar()
        
        if count > 0:
            print(f"⚠️  Zone '{zone['name']}' existe déjà, ignorée")
            continue
        
        # Insérer la zone
        insert_query = text("""
            INSERT INTO zones (id, name, region, coordinates, agro_type, created_at)
            VALUES (:id, :name, :region, CAST(:coordinates AS json), :agro_type, :created_at)
        """)
        
        import json
        conn.execute(insert_query, {
            "id": zone["id"],
            "name": zone["name"],
            "region": zone["region"],
            "coordinates": json.dumps(zone["coordinates"]),
            "agro_type": zone["agro_type"],
            "created_at": datetime.now()
        })
        
        print(f"✅ Zone créée : {zone['name']} ({zone['agro_type']})")
    
    conn.commit()

print("\n🎉 Seed terminé avec succès !")
print("\n📊 Zones disponibles dans la base :")
print("   1. Boucle du Mouhoun (GRENIER)")
print("   2. Hauts-Bassins (COTON)")
print("   3. Sahel (PASTORAL)")
print("   4. Centre-Ouest (MARAICHAGE)")
print("   5. Sud-Ouest (FRUITIER)")

print("\n🚀 Prochaine étape : Tester les agents WatcherAgent et BroadcasterAgent")
print("   → python backend/agents/watcher.py")
print("   → python backend/agents/broadcaster.py")
