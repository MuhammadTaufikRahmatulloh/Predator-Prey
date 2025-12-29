import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# STYLE / WARNA
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
h1 {
    color: #1f3c88;
    text-align: center;
}
h2, h3 {
    color: #274c77;
}
.block-container {
    padding: 2rem 3rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# JUDUL
# =========================
st.title("Simulasi Predator–Prey (Lotka–Volterra)")
st.markdown(
    "Visualisasi interaksi **Algae (Prey)** dan **Rotifers (Predator)** "
    "menggunakan model Lotka–Volterra."
)

st.markdown("---")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("C1.csv")
df.columns = df.columns.str.strip()

time = df['time (days)'].values
prey = df['algae (10^6 cells/ml)'].values
pred = df['rotifers (animals/ml)'].values

# =========================
# PARAMETER MODEL
# =========================
# ⬇️ GANTI DENGAN HASIL FITTING KAMU
alpha = 1.0
beta  = 0.1
delta = 0.1
gamma = 1.0

# =========================
# MODEL LOTKA–VOLTERRA
# =========================
def lotka_volterra(t, z):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# =========================
# SIMULASI
# =========================
sol = solve_ivp(
    lotka_volterra,
    (time.min(), time.max()),
    [prey[0], pred[0]],
    t_eval=time
)

sim_x, sim_y = sol.y

# =========================
# OVERLAY PLOT
# =========================
st.subheader("Overlay Data Asli vs Simulasi")

fig1, ax1 = plt.subplots()
ax1.plot(time, prey, 'o', color='#2a9d8f', label='Prey (Data)', alpha=0.6)
ax1.plot(time, pred, 'o', color='#e76f51', label='Predator (Data)', alpha=0.6)
ax1.plot(time, sim_x, '-', color='#264653', label='Prey (Sim)')
ax1.plot(time, sim_y, '-', color='#f4a261', label='Predator (Sim)')
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Population")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# =========================
# PHASE PORTRAIT
# =========================
st.subheader("Phase Portrait (Ruang Fase)")

fig2, ax2 = plt.subplots()
ax2.plot(sim_x, sim_y, color='#6a4c93')
ax2.set_xlabel("Prey (Algae)")
ax2.set_ylabel("Predator (Rotifers)")
ax2.grid(True)
st.pyplot(fig2)

# =========================
# INFO PARAMETER
# =========================
st.markdown("---")
st.subheader("Parameter Model")

st.write(f"**α (pertumbuhan prey)** = {alpha}")
st.write(f"**β (predasi)** = {beta}")
st.write(f"**δ (pertumbuhan predator)** = {delta}")
st.write(f"**γ (kematian predator)** = {gamma}")
