import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# STYLE / WARNA (FIX JUDUL)
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f4f6fb;
}
h1 {
    color: #1b263b;   /* WARNA JUDUL DIPERTEGAS */
    text-align: center;
    font-weight: 700;
}
h2, h3 {
    color: #415a77;
}
p {
    font-size: 16px;
    color: #1b263b;
}
.block-container {
    padding: 2.5rem 3rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# JUDUL (FIX KONTRAS)
# =========================
st.title("Simulasi Predator–Prey (Lotka–Volterra)")
st.markdown(
    "<p style='text-align:center; font-size:17px;'>"
    "Visualisasi interaksi <b>Algae (Prey)</b> dan <b>Rotifers (Predator)</b> "
    "menggunakan model Lotka–Volterra</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# LOAD DATA (AMAN)
# =========================
df = pd.read_csv("C1.csv")
df.columns = df.columns.str.strip()

df = df[
    ['time (days)', 'algae (10^6 cells/ml)', 'rotifers (animals/ml)']
].dropna()

df = df.sort_values('time (days)')

time = df['time (days)'].values
prey = df['algae (10^6 cells/ml)'].values
pred = df['rotifers (animals/ml)'].values

if len(prey) == 0 or len(pred) == 0:
    st.error("Data tidak valid setelah pembersihan NaN.")
    st.stop()

# =========================
# SIDEBAR (INTERAKTIF)
# =========================
st.sidebar.header("Pengaturan Parameter Model")

alpha = st.sidebar.slider("α (Pertumbuhan Prey)", 0.1, 2.0, 1.0, 0.1)
beta  = st.sidebar.slider("β (Predasi)", 0.01, 1.0, 0.1, 0.01)
delta = st.sidebar.slider("δ (Pertumbuhan Predator)", 0.01, 1.0, 0.1, 0.01)
gamma = st.sidebar.slider("γ (Kematian Predator)", 0.1, 2.0, 1.0, 0.1)

run = st.sidebar.button("Jalankan Simulasi")

# =========================
# MODEL LOTKA–VOLTERRA
# =========================
def lotka_volterra(t, z):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# =========================
# SIMULASI + OUTPUT
# =========================
if run:
    t_sim = np.linspace(time.min(), time.max(), 1000)

    sol = solve_ivp(
        lotka_volterra,
        (t_sim.min(), t_sim.max()),
        [prey[0], pred[0]],
        t_eval=t_sim
    )

    sim_x, sim_y = sol.y

    # =========================
    # OVERLAY PLOT
    # =========================
    st.subheader("Grafik Overlay Data Asli dan Simulasi")

    fig1, ax1 = plt.subplots()
    ax1.plot(time, prey, 'o', color='#2a9d8f', label='Prey (Data)', alpha=0.6)
    ax1.plot(time, pred, 'o', color='#e63946', label='Predator (Data)', alpha=0.6)
    ax1.plot(t_sim, sim_x, '-', color='#1d3557', linewidth=2, label='Prey (Simulasi)')
    ax1.plot(t_sim, sim_y, '-', color='#f4a261', linewidth=2, label='Predator (Simulasi)')
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Population")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.caption(
        "Gambar 1. Perbandingan data asli (titik) dan hasil simulasi "
        "model Lotka–Volterra (garis)."
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

    st.caption(
        "Gambar 2. Phase portrait sistem predator–prey "
        "yang menunjukkan dinamika siklik."
    )

    st.markdown("---")

    # =========================
    # INFO PARAMETER
    # =========================
    st.subheader("Parameter Model Lotka–Volterra")

    st.write(f"**α (pertumbuhan prey)** = {alpha}")
    st.write(f"**β (predasi)** = {beta}")
    st.write(f"**δ (pertumbuhan predator)** = {delta}")
    st.write(f"**γ (kematian predator)** = {gamma}")

else:
    st.info("Atur parameter di sidebar lalu klik **Jalankan Simulasi**.")
