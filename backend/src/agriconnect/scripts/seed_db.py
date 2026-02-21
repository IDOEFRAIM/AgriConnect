"""
Script d'initialisation PostgreSQL pour Agri-OS
Crée les tables directement via SQLAlchemy

Anciennement: init_database.py (racine)
Nouvel emplacement: scripts/seed_db.py
"""

import os
import sys
from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, DateTime, 
    Boolean, Integer, Float, JSON, ForeignKey, Text
)
from sqlalchemy.sql import func
from dotenv import load_dotenv

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger .env
load_dotenv()

# URL de connexion depuis .env
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/agriconnect")

print("🔌 Connexion à PostgreSQL DigitalOcean...")
engine = create_engine(DATABASE_URL, echo=True)
metadata = MetaData()

# ============================================
# 1. GESTION GÉOGRAPHIQUE (ZONING)
# ============================================
zones = Table(
    'zones', metadata,
    Column('id', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('region', String, nullable=False),
    Column('coordinates', JSON),
    Column('agro_type', String, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('updated_at', DateTime(timezone=True), onupdate=func.now())
)

# ============================================
# 2. UTILISATEURS (Producteurs)
# ============================================
users = Table(
    'users', metadata,
    Column('id', String, primary_key=True),
    Column('phone', String, unique=True, nullable=False),
    Column('name', String),
    Column('language', String, default='fr'),
    Column('zone_id', String, ForeignKey('zones.id')),
    Column('is_onboarded', Boolean, default=False),
    Column('voice_preference', String, default='azure_neural'),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('updated_at', DateTime(timezone=True), onupdate=func.now())
)

# ============================================
# 3. CULTURES UTILISATEURS
# ============================================
user_crops = Table(
    'user_crops', metadata,
    Column('id', String, primary_key=True),
    Column('user_id', String, ForeignKey('users.id', ondelete='CASCADE')),
    Column('crop_name', String, nullable=False),
    Column('surface_ha', Float),
    Column('planting_date', DateTime(timezone=True)),
    Column('expected_harvest', DateTime(timezone=True)),
    Column('status', String, default='ACTIVE'),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# ============================================
# 4. ALERTES (Event-Driven Core)
# ============================================
alerts = Table(
    'alerts', metadata,
    Column('id', String, primary_key=True),
    Column('type', String, nullable=False),
    Column('severity', String, nullable=False),
    Column('title', String, nullable=False),
    Column('message', Text, nullable=False),
    Column('zone_id', String, ForeignKey('zones.id')),
    Column('target_crops', JSON),
    Column('processed', Boolean, default=False),
    Column('broadcast_count', Integer, default=0),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('processed_at', DateTime(timezone=True))
)

# ============================================
# 5. MARCHÉ AGRICOLE
# ============================================
market_items = Table(
    'market_items', metadata,
    Column('id', String, primary_key=True),
    Column('product_name', String, nullable=False),
    Column('price_kg', Float, nullable=False),
    Column('currency', String, default='XOF'),
    Column('zone_id', String, ForeignKey('zones.id')),
    Column('market_name', String),
    Column('source', String),
    Column('date', DateTime(timezone=True), nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# ============================================
# 6. DONNÉES MÉTÉO
# ============================================
weather_data = Table(
    'weather_data', metadata,
    Column('id', String, primary_key=True),
    Column('zone_id', String, ForeignKey('zones.id')),
    Column('temperature', Float),
    Column('precipitation', Float),
    Column('humidity', Float),
    Column('wind_speed', Float),
    Column('forecast_date', DateTime(timezone=True)),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# ============================================
# 7. CONVERSATIONS
# ============================================
conversations = Table(
    'conversations', metadata,
    Column('id', String, primary_key=True),
    Column('user_id', String, ForeignKey('users.id', ondelete='CASCADE')),
    Column('channel', String, default='whatsapp'),
    Column('status', String, default='ACTIVE'),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('updated_at', DateTime(timezone=True), onupdate=func.now())
)

conversation_messages = Table(
    'conversation_messages', metadata,
    Column('id', String, primary_key=True),
    Column('conversation_id', String, ForeignKey('conversations.id', ondelete='CASCADE')),
    Column('role', String, nullable=False),
    Column('content', Text, nullable=False),
    Column('audio_url', String),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# ============================================
# 8. RAPPELS PLANIFIÉS
# ============================================
reminders = Table(
    'reminders', metadata,
    Column('id', String, primary_key=True),
    Column('user_id', String, ForeignKey('users.id', ondelete='CASCADE')),
    Column('title', String, nullable=False),
    Column('message', Text, nullable=False),
    Column('scheduled_at', DateTime(timezone=True), nullable=False),
    Column('sent', Boolean, default=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# Créer toutes les tables
print("\n🏗️  Création des tables...")
try:
    metadata.create_all(engine)
    print("\n✅ Toutes les tables ont été créées avec succès !")
    print("\n📋 Tables créées :")
    print("   1. zones (gestion géographique)")
    print("   2. users (producteurs)")
    print("   3. user_crops (cultures)")
    print("   4. alerts (alertes event-driven)")
    print("   5. market_items (marché)")
    print("   6. weather_data (météo)")
    print("   7. conversations + conversation_messages")
    print("   8. reminders (rappels)")
    
except Exception as e:
    print(f"\n❌ Erreur : {e}")

print("\n🎯 Prochaine étape : Exécuter seed_zones.py pour initialiser les zones")
