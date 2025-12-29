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
    background-color: #f4f6fb;
}
h1 {
    color: #1d3557;
    text-align: center;
}
h2, h3 {
    color: #457b9d;
}
p {
    font-size: 16px;
}
.block-container {
    padding: 2.5rem 3rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# JUDUL
# =========================
st.title("Simulasi Predator–Prey (Lotka–Volterra)")
st.markdown(
    "<p style='text-align:center;'>"
    "Visualisasi interaksi <b>Algae (Prey)</b> dan <b>Rotifers (Predator)</b> "
    "menggunakan model Lotka–Volterra</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# LOAD DATA 
# =========================
df = pd.read_csv("C1.csv")
df.columns = df.columns.str.strip()

df = df[
    ['time (days)', 'algae (10^6 cells/ml)', 'rotifers (animals/ml)']
].dropna()

time = df['time (days)'].values
prey = df['algae (10^6 cells/ml)'].values
pred = df['rotifers (animals/ml)'].values

if len(prey) == 0 or len(pred) == 0:
    st.error("Data tidak valid setelah pembersihan NaN.")
    st.stop()

# =========================
# PARAMETER MODEL
# =========================

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
st.subheader("Grafik Overlay Data Asli dan Simulasi")

fig1, ax1 = plt.subplots()
ax1.plot(time, prey, 'o', color='#2a9d8f', label='Prey (Data)', alpha=0.6)
ax1.plot(time, pred, 'o', color='#e63946', label='Predator (Data)', alpha=0.6)
ax1.plot(time, sim_x, '-', color='#1d3557', linewidth=2, label='Prey (Simulasi)')
ax1.plot(time, sim_y, '-', color='#f4a261', linewidth=2, label='Predator (Simulasi)')
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Population")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Caption overlay
st.caption(
    "Gambar 1. Perbandingan data asli (titik) dan hasil simulasi "
    "model Lotka–Volterra (garis). Terlihat bahwa populasi prey "
    "meningkat terlebih dahulu, kemudian diikuti oleh populasi predator."
)

st.markdown("---")

# =========================
# PHASE PORTRAIT
# =========================
st.subheader("Phase Portrait (Ruang Fase)")

fig2, ax2 = plt.subplots()
ax2.plot(sim_x, sim_y, color='#6a4c93', linewidth=2)
ax2.set_xlabel("Prey (Algae)")
ax2.set_ylabel("Predator (Rotifers)")
ax2.grid(True)
st.pyplot(fig2)

# Caption phase portrait
st.caption(
    "Gambar 2. Phase portrait sistem predator–prey yang menunjukkan "
    "lintasan tertutup, menandakan adanya dinamika siklik antara "
    "populasi prey dan predator."
)

# =========================
# INFO PARAMETER
# =========================
st.markdown("---")
st.subheader("Parameter Model Lotka–Volterra")

st.write(f"**α (laju pertumbuhan prey)** = {alpha}")
st.write(f"**β (laju predasi)** = {beta}")
st.write(f"**δ (laju pertumbuhan predator)** = {delta}")
st.write(f"**γ (laju kematian predator)** = {gamma}")
