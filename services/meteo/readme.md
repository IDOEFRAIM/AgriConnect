# Exemple dans votre script d'exécution ou un Agent:

# Importation :
# from services.monthly_climate_service import MonthlyClimateService
# from services.agri_bulletin_service import AgriBulletinService

# Initialisation (une seule fois au démarrage)
climate_service = MonthlyClimateService()
bulletin_service = AgriBulletinService()

# Utilisation par un Agent
historical_data = climate_service.get_historical_monthly_data(ville_id="BGL")
latest_agro_alerts = bulletin_service.get_latest_bulletins(limit=3)

# L'Agent Météo peut maintenant interpréter ces données claires.