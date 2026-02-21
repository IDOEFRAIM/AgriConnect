"""
Script de vérification de la base de données PostgreSQL

Anciennement: verify_database.py (racine)
Nouvel emplacement: scripts/verify_database.py
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/agriconnect")

engine = create_engine(DATABASE_URL)
conn = engine.connect()

print("=" * 60)
print("🗄️  VÉRIFICATION BASE DE DONNÉES AGRI-OS")
print("=" * 60)

# Compter les tables
result = conn.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
table_count = result.scalar()
print(f"\n✅ Tables créées : {table_count}")

# Lister les tables
result = conn.execute(text("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name
"""))
print("\n📋 Liste des tables :")
for row in result:
    print(f"   - {row[0]}")

# Vérifier les zones
result = conn.execute(text("SELECT COUNT(*) FROM zones"))
zone_count = result.scalar()
print(f"\n🌍 Zones agro-écologiques : {zone_count}")

if zone_count > 0:
    result = conn.execute(text("SELECT name, agro_type, region FROM zones"))
    print("\n📍 Détails des zones :")
    for row in result:
        print(f"   - {row[0]} ({row[1]}) - Région: {row[2]}")

# Vérifier les autres tables
tables_to_check = ['users', 'alerts', 'market_items', 'weather_data', 'conversations']
print("\n📊 État des autres tables :")
for table in tables_to_check:
    try:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count = result.scalar()
        print(f"   - {table}: {count} entrées")
    except Exception as e:
        print(f"   - {table}: ❌ Erreur ({e})")

print("\n" + "=" * 60)
print("✅ PostgreSQL configuré et opérationnel !")
print("=" * 60)

print("\n🚀 PROCHAINES ÉTAPES :")
print("   1. Installer dépendances voice : pip install -r requirements_agrios.txt")
print("   2. Tester WatcherAgent : python backend/agents/watcher.py")
print("   3. Tester BroadcasterAgent : python backend/agents/broadcaster.py")
print("   4. Configurer Twilio WhatsApp + Azure Speech")

conn.close()
