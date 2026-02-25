import requests
import json
import os
import time
from datetime import datetime
import sys

# Ajout du chemin racine pour permettre l'import de backend.tools
sys.path.append(os.getcwd())
from agriconnect.tools.db_handler import DBHandler

class WeatherCronScraper:
    def __init__(self):
        # Initialisation DB Handler
        self.db = DBHandler()
        
        # Utilisation de l'API Forecast pour les données actuelles et prévisions
        self.url = "https://api.open-meteo.com/v1/forecast"
        self.output_dir = "backend/sources/raw_data/meteo"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Paramètres pour obtenir des données agronomiques pertinentes
        self.params_template = {
            "current_weather": "true",
            "hourly": "temperature_2m,relativehumidity_2m,rain,soil_moisture_0_to_1cm,soil_moisture_3_to_9cm",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration,precipitation_hours,weathercode",
            "timezone": "auto",
            "past_days": 7,  # On garde une semaine d'historique immédiat
            "forecast_days": 7 # Et une semaine de prévisions
        }

    @staticmethod
    def get_representative_zones():
        # Mêmes zones que SoilGrid pour cohérence des données
        return {
            "Zone_Maraichere_Loumbila": (12.4833, -1.3833),    
            "Bassin_Coton_Ouest": (11.4000, -4.0000),          
            "Verger_Kou_Bobo": (11.2333, -4.4333),             
            "Pole_Agro_Industriel_Sud": (10.6400, -4.7600),    
            "Zone_Cerealiere_Ouest": (12.2500, -2.3627),       
            "Bassin_Soja_Est": (12.0600, 0.3600),              
            "Resilience_Sahelienne_Nord": (14.0300, -0.0300)   
        }

    def fetch_weather(self, zone_name, lat, lon):
        params = self.params_template.copy()
        params["latitude"] = lat
        params["longitude"] = lon
        
        try:
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Enrichissement des métadonnées
            final_data = {
                "metadata": {
                    "zone_name": zone_name,
                    "latitude": lat,
                    "longitude": lon,
                    "download_date": datetime.now().isoformat(),
                    "source": "OpenMeteo API",
                    "description": "Données météo agrégées (passé récent 7j + prévision 7j)"
                },
                "current": data.get("current_weather", {}),
                "daily_units": data.get("daily_units", {}),
                "daily": data.get("daily", {}),
                # On ne stocke pas tout le hourly pour limiter la taille, 
                # ou on pourrait faire une synthèse si nécessaire. 
                # OpenMeteo renvoie tout en colonnes.
            }
            return final_data
            
        except Exception as e:
            print(f"      ❌ Erreur API OpenMeteo pour {zone_name}: {e}")
            return None

    def run(self):
        print("🌤️ Démarrage du moissonnage Météo Agro-Ecologique...")
        zones = self.get_representative_zones()
        
        success_count = 0
        for zone_name, (lat, lon) in zones.items():
            print(f"📡 Récupération : {zone_name} ({lat}, {lon})...")
            
            weather_data = self.fetch_weather(zone_name, lat, lon)
            
            if weather_data:
                filename = f"{zone_name.lower()}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(weather_data, f, ensure_ascii=False, indent=4)
                
                print(f"   ✅ JSON sauvegardé : {filename}")
                
                # Push en Base de données
                # weather_data["daily"] contient les arrays (time, temp_max...) alignés
                self.db.save_weather_data(zone_name, lat, lon, weather_data.get("daily", {}))
                
                success_count += 1
            
            # Pause pour respecter les rate limits (OpenMeteo est généreux mais restons polis)
            time.sleep(1.5)
            
        print(f"🎉 Terminé. {success_count}/{len(zones)} zones mises à jour.")

if __name__ == "__main__":
    scraper = WeatherCronScraper()
    scraper.run()
