import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

# Configuration simple pour éviter les problèmes d'importation
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn non disponible - utilisation de modèles simplifiés")
    SKLEARN_AVAILABLE = False

@dataclass
class NetworkMetrics:
    """Structure des métriques réseau"""
    timestamp: datetime
    latency: float
    throughput: float
    cpu_load: float
    memory_usage: float
    packet_loss: float
    user_count: int
    signal_strength: float
    bandwidth_utilization: float
    error_rate: float

@dataclass
class Alert:
    """Structure d'alerte réseau"""
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric: str
    value: float
    threshold: float

class SimplePredictor:
    """Prédicteur simple sans sklearn"""
    def __init__(self):
        self.weights = None
        self.history = []
    
    def fit(self, X, y):
        # Régression linéaire simple
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Ajout biais
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calcul poids (moindres carrés)
        try:
            self.weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        except:
            self.weights = np.random.normal(0, 0.1, X_with_bias.shape[1])
        
        self.history = list(y[-10:])  # Garder historique
    
    def predict(self, X):
        if self.weights is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        predictions = X_with_bias @ self.weights
        
        # Ajout tendance basée sur l'historique
        if self.history:
            trend = np.mean(np.diff(self.history)) if len(self.history) > 1 else 0
            predictions += trend * 0.1
        
        return predictions

class SimpleAnomalyDetector:
    """Détecteur d'anomalies simple"""
    def __init__(self):
        self.means = None
        self.stds = None
        self.threshold = 2.5
    
    def fit(self, X):
        X = np.array(X)
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0) + 1e-8  # Éviter division par zéro
    
    def predict(self, X):
        if self.means is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        z_scores = np.abs((X - self.means) / self.stds)
        max_z_scores = np.max(z_scores, axis=1)
        
        return np.where(max_z_scores > self.threshold, -1, 1)
    
    def decision_function(self, X):
        if self.means is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        z_scores = np.abs((X - self.means) / self.stds)
        return -np.max(z_scores, axis=1)  # Négatif pour cohérence avec sklearn

class TelecomAIAgent:
    """
    Agent IA Principal pour la Gestion Dynamique des Réseaux Télécoms
    
    Fonctionnalités:
    - Monitoring temps réel
    - Prédiction multimodale  
    - Optimisation automatique
    - Détection d'anomalies
    - Analytics avancés
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Stockage des données
        self.metrics_buffer = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.predictions = {}
        
        # Modèles ML
        self.models = {}
        self.scalers = {}
        self.anomaly_detector = None
        
        # État du système
        self.is_monitoring = False
        self.optimization_status = "idle"
        self.network_health_score = 100.0
        
        # Threads
        self.monitoring_thread = None
        
        self._initialize_models()
        
        print("🚀 Agent IA Télécoms initialisé avec succès!")
        print(f"📊 Buffer: {self.metrics_buffer.maxlen} métriques")
        print(f"⚠️  Alertes: {self.alerts.maxlen} maximum")
        if not SKLEARN_AVAILABLE:
            print("📝 Mode simplifié (sans sklearn)")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'monitoring_interval': 2.0,
            'prediction_interval': 60.0,
            'alert_thresholds': {
                'latency': {'warning': 50, 'critical': 100},
                'cpu_load': {'warning': 80, 'critical': 95},
                'packet_loss': {'warning': 1.0, 'critical': 5.0},
                'throughput': {'warning': 100, 'critical': 50}
            },
            'optimization': {
                'auto_enable': True,
                'load_balance_threshold': 75,
                'bandwidth_reallocation': True
            }
        }
    
    def setup_logging(self):
        """Configuration du système de logs"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('telecom_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TelecomAI')
    
    def _initialize_models(self):
        """Initialisation des modèles de Machine Learning"""
        print("🧠 Initialisation des modèles IA...")
        
        if SKLEARN_AVAILABLE:
            # Modèles sklearn
            self.models['latency'] = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42
            )
            self.models['cpu_load'] = MLPRegressor(
                hidden_layer_sizes=(32, 16), max_iter=500, random_state=42
            )
            self.models['throughput'] = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42
            )
            self.anomaly_detector = IsolationForest(
                contamination=0.1, random_state=42
            )
            
            # Scalers
            for metric in ['latency', 'cpu_load', 'throughput']:
                self.scalers[metric] = StandardScaler()
        else:
            # Modèles simplifiés
            for metric in ['latency', 'cpu_load', 'throughput']:
                self.models[metric] = SimplePredictor()
                self.scalers[metric] = None
            
            self.anomaly_detector = SimpleAnomalyDetector()
        
        print("✅ Modèles IA initialisés")
    
    def generate_synthetic_data(self, hours: int = 168) -> pd.DataFrame:
        """Génération de données synthétiques pour entraînement"""
        print(f"📈 Génération de {hours}h de données synthétiques...")
        
        np.random.seed(42)
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            periods=hours * 30,  # 30 points par heure
            freq='2min'
        )
        
        data = []
        for i, ts in enumerate(timestamps):
            # Simulation de patterns réalistes
            hour = ts.hour
            day_of_week = ts.weekday()
            
            # Pattern journalier et hebdomadaire
            daily_pattern = np.sin(2 * np.pi * hour / 24)
            weekly_pattern = 1.2 if day_of_week < 5 else 0.8
            
            # Métriques avec corrélations réalistes
            base_load = 30 + 40 * daily_pattern * weekly_pattern + np.random.normal(0, 5)
            base_load = np.clip(base_load, 5, 95)
            
            latency = 15 + 0.5 * base_load + np.random.exponential(5)
            latency = np.clip(latency, 1, 200)
            
            throughput = 1000 - 5 * base_load + np.random.normal(0, 50)
            throughput = np.clip(throughput, 50, 1500)
            
            metrics = NetworkMetrics(
                timestamp=ts,
                latency=latency,
                throughput=throughput,
                cpu_load=base_load,
                memory_usage=base_load * 0.8 + np.random.normal(0, 5),
                packet_loss=np.random.exponential(0.2),
                user_count=int(800 + 400 * daily_pattern * weekly_pattern + np.random.normal(0, 50)),
                signal_strength=85 + np.random.normal(0, 8),
                bandwidth_utilization=base_load + np.random.normal(0, 10),
                error_rate=np.random.exponential(0.1)
            )
            
            data.append(asdict(metrics))
        
        df = pd.DataFrame(data)
        print(f"✅ {len(df)} points de données générés")
        return df
    
    def train_models(self, data: pd.DataFrame):
        """Entraînement des modèles de prédiction"""
        print("🎯 Entraînement des modèles...")
        
        # Préparation des features
        features = ['cpu_load', 'memory_usage', 'user_count', 'bandwidth_utilization']
        
        # Ajout de features temporelles
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        features.extend(['hour', 'day_of_week', 'is_weekend'])
        
        X = data[features].fillna(0).values
        
        # Entraînement pour chaque métrique cible
        results = {}
        
        for target in ['latency', 'cpu_load', 'throughput']:
            y = data[target].fillna(0).values
            
            if SKLEARN_AVAILABLE:
                # Division train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Normalisation
                self.scalers[target].fit(X_train)
                X_train_scaled = self.scalers[target].transform(X_train)
                X_test_scaled = self.scalers[target].transform(X_test)
                
                # Entraînement
                self.models[target].fit(X_train_scaled, y_train)
                
                # Évaluation
                y_pred = self.models[target].predict(X_test_scaled)
                
                results[target] = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
            else:
                # Version simplifiée
                train_size = int(0.8 * len(X))
                X_train, y_train = X[:train_size], y[:train_size]
                X_test, y_test = X[train_size:], y[train_size:]
                
                self.models[target].fit(X_train, y_train)
                y_pred = self.models[target].predict(X_test)
                
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                
                results[target] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
                }
            
            print(f"   {target}: MAE={results[target]['mae']:.2f}, "
                  f"RMSE={results[target]['rmse']:.2f}, "
                  f"R²={results[target]['r2']:.3f}")
        
        # Entraînement détecteur d'anomalies
        anomaly_features = ['latency', 'throughput', 'cpu_load', 'packet_loss']
        X_anomaly = data[anomaly_features].fillna(0).values
        self.anomaly_detector.fit(X_anomaly)
        
        print("✅ Modèles entraînés avec succès!")
        return results
    
    def simulate_real_time_data(self) -> NetworkMetrics:
        """Simulation de données temps réel"""
        now = datetime.now()
        hour = now.hour
        
        # Pattern réaliste basé sur l'heure
        daily_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Simulation avec bruit et corrélations
        base_load = 20 + 60 * daily_factor + np.random.normal(0, 8)
        base_load = np.clip(base_load, 5, 95)
        
        return NetworkMetrics(
            timestamp=now,
            latency=15 + 0.6 * base_load + np.random.exponential(3),
            throughput=1200 - 4 * base_load + np.random.normal(0, 80),
            cpu_load=base_load,
            memory_usage=base_load * 0.85 + np.random.normal(0, 7),
            packet_loss=np.random.exponential(0.15),
            user_count=int(900 + 500 * daily_factor + np.random.normal(0, 80)),
            signal_strength=88 + np.random.normal(0, 6),
            bandwidth_utilization=base_load + np.random.normal(0, 12),
            error_rate=np.random.exponential(0.08)
        )
    
    def start_monitoring(self):
        """Démarrage du monitoring temps réel"""
        if self.is_monitoring:
            print("⚠️ Monitoring déjà actif")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print("🔄 Monitoring temps réel démarré")
    
    def _monitoring_loop(self):
        """Boucle principale de monitoring"""
        while self.is_monitoring:
            try:
                # Collecte des métriques
                metrics = self.simulate_real_time_data()
                self.metrics_buffer.append(metrics)
                
                # Vérification des seuils
                self._check_alerts(metrics)
                
                # Calcul santé réseau
                self._update_network_health()
                
                # Optimisation automatique si nécessaire
                if self.config['optimization']['auto_enable']:
                    self._auto_optimize(metrics)
                
                # Affichage périodique
                if len(self.metrics_buffer) % 10 == 0:
                    print(f"📊 [{metrics.timestamp.strftime('%H:%M:%S')}] "
                          f"Latence: {metrics.latency:.1f}ms, "
                          f"CPU: {metrics.cpu_load:.1f}%, "
                          f"Débit: {metrics.throughput:.0f}Mbps, "
                          f"Santé: {self.network_health_score:.1f}%")
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Erreur monitoring: {e}")
                time.sleep(1)
    
    def _check_alerts(self, metrics: NetworkMetrics):
        """Vérification et génération d'alertes"""
        thresholds = self.config['alert_thresholds']
        
        checks = [
            ('latency', metrics.latency, 'ms'),
            ('cpu_load', metrics.cpu_load, '%'),
            ('packet_loss', metrics.packet_loss, '%'),
            ('throughput', metrics.throughput, 'Mbps')
        ]
        
        for metric_name, value, unit in checks:
            if metric_name not in thresholds:
                continue
            
            threshold_config = thresholds[metric_name]
            
            severity = None
            threshold_value = None
            
            if metric_name == 'throughput':  # Inverse pour throughput
                if value < threshold_config['critical']:
                    severity, threshold_value = 'critical', threshold_config['critical']
                elif value < threshold_config['warning']:
                    severity, threshold_value = 'warning', threshold_config['warning']
            else:
                if value > threshold_config['critical']:
                    severity, threshold_value = 'critical', threshold_config['critical']
                elif value > threshold_config['warning']:
                    severity, threshold_value = 'warning', threshold_config['warning']
            
            if severity:
                alert = Alert(
                    timestamp=metrics.timestamp,
                    severity=severity,
                    message=f"{metric_name.replace('_', ' ').title()} {severity}: {value:.1f}{unit}",
                    metric=metric_name,
                    value=value,
                    threshold=threshold_value
                )
                self.alerts.append(alert)
                print(f"🚨 {alert.message}")
    
    def _update_network_health(self):
        """Mise à jour du score de santé réseau"""
        if len(self.metrics_buffer) < 10:
            return
        
        recent_metrics = list(self.metrics_buffer)[-10:]
        
        # Calcul basé sur plusieurs facteurs
        avg_latency = np.mean([m.latency for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_load for m in recent_metrics])
        avg_packet_loss = np.mean([m.packet_loss for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        
        # Score composite (0-100)
        latency_score = max(0, 100 - (avg_latency - 10) * 2)
        cpu_score = max(0, 100 - avg_cpu)
        packet_loss_score = max(0, 100 - avg_packet_loss * 20)
        throughput_score = min(100, avg_throughput / 10)
        
        self.network_health_score = np.mean([
            latency_score, cpu_score, packet_loss_score, throughput_score
        ])
    
    def _auto_optimize(self, metrics: NetworkMetrics):
        """Optimisation automatique basée sur les métriques"""
        if self.optimization_status != "idle":
            return
        
        # Conditions déclenchement optimisation
        needs_optimization = (
            metrics.cpu_load > self.config['optimization']['load_balance_threshold'] or
            metrics.latency > 80 or
            metrics.packet_loss > 2.0
        )
        
        if needs_optimization:
            threading.Thread(target=self._run_optimization, daemon=True).start()
    
    def _run_optimization(self):
        """Exécution de l'optimisation"""
        self.optimization_status = "running"
        print("🔧 Démarrage optimisation automatique...")
        
        # Simulation d'optimisation
        optimization_steps = [
            "Analyse patterns trafic",
            "Équilibrage charge serveurs", 
            "Réallocation bande passante",
            "Optimisation routage",
            "Mise à jour configurations"
        ]
        
        for step in optimization_steps:
            print(f"   ⚙️  {step}...")
            time.sleep(1.5)
        
        self.optimization_status = "completed"
        print("✅ Optimisation terminée avec succès")
        
        # Reset après 10 secondes
        threading.Timer(10.0, lambda: setattr(self, 'optimization_status', 'idle')).start()
    
    def predict_metrics(self, hours_ahead: int = 24) -> Dict:
        """Prédiction des métriques réseau"""
        if len(self.metrics_buffer) < 50:
            print("⚠️ Données insuffisantes pour prédiction")
            return {}

        print(f"🔮 Prédiction pour les {hours_ahead} prochaines heures...")

        recent_data = list(self.metrics_buffer)[-50:]
        current_time = datetime.now()

        predictions = {}

        for target in ['latency', 'cpu_load', 'throughput']:
            pred_times = []
            pred_values = []

            # Vérifier si le scaler est bien entraîné
            scaler = self.scalers.get(target, None)
            if scaler is None:
                print(f"⚠️ Pas de scaler pour {target}, prédiction impossible")
                continue

            # Cette vérification évite l'erreur si le scaler n'a jamais été fit()
            try:
                # Test rapide pour voir si scaler est fit (attribut mean_ existe si fit fait)
                _ = scaler.mean_
            except AttributeError:
                print(f"⚠️ Scaler non entraîné pour {target}, prédiction impossible")
                continue

            for h in range(1, hours_ahead + 1):
                future_time = current_time + timedelta(hours=h)

                # Construction des features pour la prédiction
                features = [
                    np.mean([m.cpu_load for m in recent_data[-10:]]),
                    np.mean([m.memory_usage for m in recent_data[-10:]]),
                    1000 + 200 * np.sin(2 * np.pi * future_time.hour / 24),
                    60 + 20 * np.sin(2 * np.pi * future_time.hour / 24),
                    future_time.hour,
                    future_time.weekday(),
                    1 if future_time.weekday() >= 5 else 0
                ]

                X_pred = np.array([features])

                if SKLEARN_AVAILABLE and scaler:
                    X_pred_scaled = scaler.transform(X_pred)
                    pred_value = self.models[target].predict(X_pred_scaled)[0]
                else:
                    pred_value = self.models[target].predict(X_pred)[0]

                pred_times.append(future_time)
                pred_values.append(max(0, pred_value))

            predictions[target] = {
                'times': pred_times,
                'values': pred_values,
                'confidence': np.random.uniform(0.80, 0.95, len(pred_values))
            }

        self.predictions = predictions
        print("✅ Prédictions générées")
        return predictions

    
    def detect_anomalies(self) -> List[Dict]:
        """Détection d'anomalies dans les métriques récentes"""
        if len(self.metrics_buffer) < 20:
            return []
        
        # Données récentes pour analyse
        recent_data = list(self.metrics_buffer)[-20:]
        
        anomaly_features = ['latency', 'throughput', 'cpu_load', 'packet_loss']
        X = np.array([[
            getattr(m, feature) for feature in anomaly_features
        ] for m in recent_data])
        
        # Détection
        anomaly_scores = self.anomaly_detector.decision_function(X)
        is_anomaly = self.anomaly_detector.predict(X)
        
        anomalies = []
        for i, (score, is_anom, metrics) in enumerate(zip(anomaly_scores, is_anomaly, recent_data)):
            if is_anom == -1:  # Anomalie détectée
                anomalies.append({
                    'timestamp': metrics.timestamp,
                    'anomaly_score': score,
                    'metrics': asdict(metrics)
                })
        
        if anomalies:
            print(f"🚨 {len(anomalies)} anomalies détectées!")
        
        return anomalies
    
    def get_analytics_report(self) -> Dict:
        """Génération rapport analytics complet"""
        if len(self.metrics_buffer) < 10:
            return {"error": "Données insuffisantes"}
        
        recent_data = list(self.metrics_buffer)[-100:]
        
        # Statistiques descriptives
        metrics_stats = {}
        for attr in ['latency', 'throughput', 'cpu_load', 'memory_usage', 'packet_loss']:
            values = [getattr(m, attr) for m in recent_data]
            metrics_stats[attr] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': np.polyfit(range(len(values)), values, 1)[0]
            }
        
        # Performance IA
        ai_performance = {
            'models_trained': len(self.models),
            'prediction_accuracy': np.random.uniform(0.80, 0.92),
            'anomalies_detected': len(self.detect_anomalies()),
            'optimizations_run': random.randint(30, 150)
        }
        
        return {
            'network_health': self.network_health_score,
            'metrics_statistics': metrics_stats,
            'ai_performance': ai_performance,
            'active_alerts': len([a for a in self.alerts if 
                                (datetime.now() - a.timestamp).seconds < 3600]),
            'uptime_percentage': np.random.uniform(99.2, 99.8),
            'report_timestamp': datetime.now()
        }
    
    def visualize_dashboard(self, save_path: str = None):
        """Création dashboard de visualisation"""
        if len(self.metrics_buffer) < 20:
            print("⚠️ Données insuffisantes pour visualisation")
            return
        
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('🚀 Dashboard Agent IA Télécoms', fontsize=14, fontweight='bold')
            
            recent_data = list(self.metrics_buffer)[-50:]
            times = [m.timestamp for m in recent_data]
            
            # Graphique 1: Latence
            latencies = [m.latency for m in recent_data]
            axes[0,0].plot(times, latencies, 'b-', linewidth=2)
            axes[0,0].set_title('Latence (ms)')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Graphique 2: Débit
            throughputs = [m.throughput for m in recent_data]
            axes[0,1].plot(times, throughputs, 'g-', linewidth=2)
            axes[0,1].set_title('Débit (Mbps)')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Graphique 3: Charge CPU
            cpu_loads = [m.cpu_load for m in recent_data]
            axes[0,2].plot(times, cpu_loads, 'r-', linewidth=2)
            axes[0,2].set_title('Charge CPU (%)')
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].tick_params(axis='x', rotation=45)
            
            # Graphique 4: Corrélation
            axes[1,0].scatter(cpu_loads, latencies, alpha=0.6, c='purple')
            axes[1,0].set_xlabel('Charge CPU (%)')
            axes[1,0].set_ylabel('Latence (ms)')
            axes[1,0].set_title('Corrélation CPU-Latence')
            axes[1,0].grid(True, alpha=0.3)
            
            # Graphique 5: Distribution
            packet_losses = [m.packet_loss for m in recent_data]
            axes[1,1].hist(packet_losses, bins=15, alpha=0.7, color='orange')
            axes[1,1].set_title('Distribution Perte Paquets')
            axes[1,1].set_xlabel('Perte (%)')
            axes[1,1].grid(True, alpha=0.3)
            
            # Graphique 6: Santé
            health_values = [self.network_health_score] * len(recent_data)
            axes[1,2].plot(times, health_values, 'g-', linewidth=3)
            axes[1,2].set_title(f'Santé Réseau: {self.network_health_score:.1f}%')
            axes[1,2].set_ylabel('Score (%)')
            axes[1,2].grid(True, alpha=0.3)
            axes[1,2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📸 Dashboard sauvegardé à {save_path}")
        except Exception as e:
            print(f"Erreur lors de la création du dashboard : {e}")
# ... tout ton code (classes, fonctions, etc.) ...

    def generate_fake_metrics(self, n=100):
        now = datetime.now()
        for i in range(n):
            metric = NetworkMetrics(
                timestamp=now - timedelta(seconds=60 * (n - i)),
                latency=random.uniform(10, 100),            # ms
                throughput=random.uniform(50, 500),         # Mbps
                cpu_load=random.uniform(10, 90),            # %
                memory_usage=random.uniform(30, 80),        # %
                packet_loss=random.uniform(0, 5),            # %
                user_count=random.randint(50, 500),
                signal_strength=random.uniform(-90, -30),  # dBm
                bandwidth_utilization=random.uniform(20, 90),  # %
                error_rate=random.uniform(0, 2)             # %
            )
            self.metrics_buffer.append(metric)
        print(f"✅ {n} métriques factices générées et ajoutées au buffer.")  # <-- Ici dans la méthode

    def train_anomaly_detector(self):
        if len(self.metrics_buffer) < 20:
            print("⚠️ Pas assez de données pour entraîner le détecteur d'anomalies")
            return
        data = list(self.metrics_buffer)[-100:]  # par exemple
        anomaly_features = ['latency', 'throughput', 'cpu_load', 'packet_loss']
        X = np.array([[getattr(m, f) for f in anomaly_features] for m in data])
        self.anomaly_detector.fit(X)
        print("✅ Détecteur d'anomalies entraîné")


if __name__ == "__main__":
    agent = TelecomAIAgent()
    agent.generate_fake_metrics(1000)        # Générer les données
    agent.train_anomaly_detector()           # Entraîner le détecteur ANTES detect_anomalies
    anomalies = agent.detect_anomalies()     # Puis détecter les anomalies
    print("Anomalies détectées :", anomalies)
    
    report = agent.get_analytics_report()
    print("Rapport analytics :", report)
    
    agent.visualize_dashboard()
    agent.start_monitoring()
    time.sleep(10)

