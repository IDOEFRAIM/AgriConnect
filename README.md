
# 🌿 Assistant IA Contextuel – Agriculture Burkinabè (Mil)

## 🎯 Sujet choisi et justification

Nous avons choisi le **mil** comme sujet burkinabè pour son importance stratégique dans l’agriculture locale, la sécurité alimentaire et les pratiques culturales traditionnelles. Ce choix garantit :

- Une **documentation accessible** (rapports FAO, publications locales)  
- Une **pertinence directe** pour les utilisateurs burkinabè  
- Une **valeur éducative** pour les agriculteurs, étudiants et décideurs

---

## 🧠 Architecture technique

Notre système repose sur une architecture **RAG 100% open source**, conçue pour fonctionner localement sans dépendance propriétaire :

```
Question utilisateur
      ↓
Embeddings (Gemma:2b via OllamaEmbedding)
      ↓
Recherche vectorielle (ChromaDB)
      ↓
Documents pertinents (Chromadb.as_retriever())
      ↓
LLM (Gemma:2b via Ollama)
      ↓
Réponse + Sources
```

---

## 🛠️ Technologies open source utilisées

- 🧠 **LangChain** (Framework IA)  
  Licence : MIT  
  [Voir la licence](https://github.com/langchain-ai/langchain/blob/master/LICENSE)

- 🧠 **Gemma:2b via Ollama** (Embeddings & LLM)  
  Licence : Apache 2.0  
  [Voir la licence](https://www.apache.org/licenses/LICENSE-2.0)

- 📦 **ChromaDB** (Vectorstore)  
  Licence : Apache 2.0  
  [Voir la licence](https://github.com/chroma-core/chroma/blob/main/LICENSE)

- 🔧 **Flask** (Backend API)  
  Licence : BSD-3-Clause  
  [Voir la licence](https://github.com/pallets/flask/blob/main/LICENSE.rst)

- 🎛️ **Gradio** (Frontend)  
  Licence : Apache 2.0  
  [Voir la licence](https://github.com/gradio-app/gradio/blob/main/LICENSE)

- 🧹 **BeautifulSoup, LangDetect, PDFMiner** (Scraping & Traitement)  
  Licence : MIT / BSD  
  [Voir la licence](https://github.com/wention/BeautifulSoup4/blob/master/LICENSE)
# 1. Cloner le projet
git clone https://github.com/ton-utilisateur/agribot-mil.git
cd agribot-mil

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l’API backend
python src/api.py

# 4. Lancer l’interface utilisateur
python frontend/main.py
```

---

## 📊 Résultats de l’évaluation

## 📊 Performances du système

| Critère                  | Score         |
|--------------------------|---------------|
| Précision Retrieval      | 85%           |
| Pertinence des Réponses       | 4.2 / 5       |
| Temps moyen de réponse   | 1.8 sec       |
---

## 📁 Structure du projet

```
agribot/
├── data/
│   ├── corpus.json
│   └── sources.txt
├── src/
    ├──__init__.py
│   ├── data_extraction.py
│   ├── data_processing.py
│   ├── data_vectordb.py
│   └── api.py
├── frontend/
│   └── main.py
├── evaluation/
│   ├── questions.json
│   ├── resultats.json
│   └── test.py 
├── requirements.txt
├── LICENCE.md
└── README.md
```

---

## ✅ Bonus intégrés

- ✅ Déploiement en ligne via Gradio Tunnel (Cloudflare)
- ✅ Vidéo démo (YouTube)

---

## 📜 Licence

Ce projet est sous licence **MIT**, garantissant liberté d’utilisation, modification et redistribution.

---

Voici une version améliorée et bien structurée de ta section “Remerciements” et “Fonctionnalités futures”, avec une formulation fluide et inspirante pour ton `README.md` :

---

## 🙌 Remerciements

Merci à **MTDPCE** pour cette initiative visionnaire. Ce projet vise à promouvoir l’autonomie technologique, l’apprentissage collectif et l’impact local à travers l’open source. Nous croyons en une innovation accessible, éthique et adaptée aux réalités du Burkina Faso.

---

## 🚀 Fonctionnalités prévues dans les prochaines versions

Par manque de temps et de moyens, nous n’avons pu implémenter qu’une partie des fonctionnalités envisagées. Dans un futur proche, nous souhaitons :

- 🧠 **Intégrer un système de détection des maladies des plantes** à partir d’images, grâce à des modèles légers comme **EfficientNet**, capables de tourner sur des téléphones tout en conservant une excellente précision.
- 🤖 **Transformer AGRIBOT en un véritable agent IA** autonome et interactif, en exploitant des frameworks comme **LangGraph** pour gérer les dialogues, les actions et les états de manière dynamique.
- 📱 **Optimiser l’accessibilité mobile**, afin que les agriculteurs puissent bénéficier de conseils intelligents directement depuis leur smartphone, même en zone rurale.

Osons rêver. Osons rendre l’impossible possible au **Burkina Faso**.After all ,Sky is the limit.


