# Telecom-AI-Agent-Monitoring-R-seau-Intelligent

## 📌 Description
**Telecom AI Agent** est une application interactive qui utilise **l’intelligence artificielle** pour analyser et surveiller un réseau télécom en temps réel.  
Elle détecte automatiquement les **anomalies**, génère des **prévisions de performance**, et affiche un **dashboard dynamique** avec alertes et graphiques.

Ce projet est une démonstration concrète d’un **pipeline IA complet** :
1. **Collecte** de données réseau (factices ici, réelles en production)
2. **Analyse & détection d’anomalies** avec Machine Learning
3. **Prévision** de métriques réseau
4. **Visualisation** et interaction via **Streamlit**

---

## 🖥️ Fonctionnalités
- 📊 **Tableau de bord en temps réel** avec indicateurs clés
- 🚨 **Détection automatique** d’anomalies réseau (Isolation Forest)
- 🔮 **Prévision des métriques** (modèles de régression)
- 📈 **Visualisations graphiques** des tendances et anomalies
- 🛠 **Génération de données factices** pour tests et simulations
- ⚡ Interface **dynamique** mise à jour instantanément

---

## 📂 Structure du projet
tpe/
│── agent.py # Logique IA : génération de données, détection anomalies, prédiction
│── app.py # Interface utilisateur avec Streamlit
│── requirements.txt # Liste des dépendances Python
│── README.md # Documentation du projet


## ⚠️ Limites actuelles
- **Utilise des données simulées (non connectées à un vrai réseau)**
- **Modèles IA basiques (pas optimisés pour production)**
- **Pas de système d’alertes externe (email, Slack, etc.)**
- **Pas de persistance longue durée (pas de base de données)**
- 
## 📈 Améliorations possibles
- **Connexion à des données réseau réelles (SNMP, API, logs…)**
- **Optimisation et validation avancée des modèles IA**
- **Intégration d’alertes automatiques multi-canaux**
- **Ajout d’une base de données pour l’historique des métriques**
- **Interface plus personnalisable et orientée métier**



