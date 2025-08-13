# app.py
import streamlit as st
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
import time
from agent import TelecomAIAgent

# =======================
# App principale
# =======================
def main():
    st.set_page_config(page_title="🚀 Agent IA Télécoms", layout="wide")

    st.title("🚀 Agent IA Télécoms - Monitoring Réseau Temps Réel")

    # -----------------------
    # Initialisation Agent
    # -----------------------
    if "agent" not in st.session_state:
        agent = TelecomAIAgent()
        agent.generate_fake_metrics(1000)
        agent.train_anomaly_detector()
        st.session_state.agent = agent
        st.session_state.last_update = datetime.now()
        st.success("🚀 Agent IA Télécoms initialisé avec succès !")

    agent = st.session_state.agent

    # -----------------------
    # Paramètres de rafraîchissement
    # -----------------------
    st.sidebar.header("⚙️ Options")
    refresh_seconds = st.sidebar.slider("⏱ Intervalle de rafraîchissement", 1, 10, 5)
    auto_refresh = st.sidebar.checkbox("🔄 Rafraîchissement automatique", value=True)

    if st.sidebar.button("🔄 Forcer un rafraîchissement"):
        st.session_state.last_update = datetime.now()
        st.rerun()

    # -----------------------
    # Simuler de nouvelles métriques à chaque affichage
    # -----------------------
    agent.generate_fake_metrics(5)  # 5 nouvelles métriques par cycle

    # -----------------------
    # Rapport Analytics
    # -----------------------
    st.subheader("📊 Rapport Analytics Réseau")
    report = agent.get_analytics_report()
    if "error" in report:
        st.warning(report["error"])
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Santé Réseau", f"{report['network_health']:.1f}%")
        col2.metric("Alertes Actives (1h)", report['active_alerts'])
        col3.metric("Uptime %", f"{report['uptime_percentage']:.2f}%")

        st.write("### 📈 Statistiques (dernières 100 mesures)")
        for metric, stats in report['metrics_statistics'].items():
            st.write(f"**{metric.title()}**")
            st.write(f"- Moyenne : {stats['mean']:.2f}")
            st.write(f"- Écart-type : {stats['std']:.2f}")
            st.write(f"- Min : {stats['min']:.2f}")
            st.write(f"- Max : {stats['max']:.2f}")
            st.write(f"- Tendance : {stats['trend']:.4f}")

    # -----------------------
    # Alertes
    # -----------------------
    st.subheader("🚨 Alertes récentes")
    alerts = list(agent.alerts)[-20:] if hasattr(agent, "alerts") else []
    for alert in reversed(alerts):
        age_sec = (datetime.now() - alert.timestamp).seconds
        if age_sec < 3600:
            st.warning(f"{alert.timestamp.strftime('%H:%M:%S')} - {alert.message}")

    # -----------------------
    # Anomalies
    # -----------------------
    st.subheader("🔍 Anomalies détectées")
    anomalies = agent.detect_anomalies()
    if anomalies:
        for anomaly in anomalies:
            ts = anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            st.error(f"Anomalie à {ts} — score {anomaly['anomaly_score']:.2f}")
            st.json(anomaly['metrics'])
    else:
        st.info("✅ Aucune anomalie détectée récemment.")

    # -----------------------
    # Prédictions
    # -----------------------
    st.subheader("🔮 Prédictions métriques")
    hours = st.slider("Heures à prévoir", 1, 48, 24)
    if st.button("Générer prédictions"):
        try:
            preds = agent.predict_metrics(hours)
            for metric, data in preds.items():
                st.write(f"### {metric.title()}")
                fig, ax = plt.subplots()
                ax.plot(data['times'], data['values'], label=f"Prédiction {metric}")
                ax.set_xlabel("Temps")
                ax.set_ylabel(metric)
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors des prédictions : {e}")

    # -----------------------
    # Dashboard
    # -----------------------
    st.subheader("📈 Dashboard visuel")
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            tmp_path = tmpfile.name
        agent.visualize_dashboard(tmp_path)
        with open(tmp_path, "rb") as f:
            st.image(f.read(), caption="Dashboard Réseau", use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la création du dashboard : {e}")

    # -----------------------
    # Auto refresh
    # -----------------------
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
