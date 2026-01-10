# Utilisation de Python 3.13 comme spécifié dans pyproject.toml
FROM python:3.13-slim

# Métadonnées
LABEL maintainer="AgConnect Team"
LABEL description="Agriculture Multi-Agent System with Robust RAG"

# Variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.2

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système minimales
# build-essential pour compiler certaines librairies (numpy, faiss, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configuration de Poetry pour ne pas utiliser de virtualenv dans le conteneur
RUN poetry config virtualenvs.create false

# Copie des fichiers de dépendances
COPY pyproject.toml poetry.lock* /app/

# Installation des dépendances (sans les dev dependencies pour la prod)
# Note: On utilise --no-root car le code n'est pas encore copié
RUN poetry install --no-interaction --no-ansi --no-root

# Copie du code source
COPY . /app

# Installation du projet lui-même (si nécessaire pour les scripts d'entrée)
RUN poetry install --no-interaction --no-ansi

# Création des dossiers de données nécessaires
RUN mkdir -p data/vector_store logs reports

# Exposition du port API (si backend FastAPI utilisé)
EXPOSE 8000

# Commande de démarrage par défaut (Orchestrator ou API)
# Ici on lance l'API backend par défaut
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
