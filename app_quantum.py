
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


# =====================================
# CONFIGURACIÓN
# =====================================

st.set_page_config(
    page_title="Nopal Quant | QAOA",
    page_icon="🌵",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: #161b22;
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #6cb42d;
}
div[data-testid="stMetric"] {
    background-color: #244A78;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
st.image("NQH.svg", caption="", width=400)
st.title("Nopal Quant | Selección de acciones con QAOA")

st.write(
    "Modelo de decisión binaria para mantener o vender acciones "
    "mediante Quantum Approximate Optimization Algorithm."
)


# =====================================
# PARÁMETROS
# =====================================

st.sidebar.header("Parámetros del modelo")

tickers_input = st.sidebar.text_area(
    "Claves de acciones (Yahoo Finance)",
    value="BIMBOA.MX,WALMEX.MX,ALSEA.MX,GCARSOA1.MX,FEMSAUBD.MX"
)

period = st.sidebar.selectbox(
    "Periodo histórico",
    ["1y", "2y", "5y"],
    index=1
)

lambda_risk = st.sidebar.number_input(
    "Aversión al riesgo (λ)",
    min_value=0.0,
    max_value=10.0,
    value=0.02,
    step=0.01
)

reps = st.sidebar.slider(
    "Profundidad QAOA (p)",
    min_value=1,
    max_value=5,
    value=2
)

maxiter = st.sidebar.slider(
    "Iteraciones COBYLA",
    min_value=10,
    max_value=500,
    value=150
)

shots = st.sidebar.number_input(
    "Número de shots",
    min_value=100,
    max_value=10000,
    value=1024,
    step=100
)

ejecutar = st.sidebar.button(
    "Ejecutar optimización",
    type="primary"
)


# =====================================
# EJECUCIÓN
# =====================================

if ejecutar:

    tickers = [
        t.strip().upper()
        for t in tickers_input.split(",")
        if t.strip()
    ]

    if len(tickers) < 2:
        st.error("Introduce al menos dos acciones.")

        st.stop()

    try:

        with st.spinner("Descargando información financiera..."):

            # =====================================
            # 1. DATOS DE MERCADO
            # =====================================

            data = yf.download(
                tickers,
                period=period,
                auto_adjust=True,
                progress=False
            )["Close"]

            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])

            data = data.dropna(axis=1, how="all")

            # Conservar únicamente activos con información suficiente
            data = data.dropna()

            if data.empty or data.shape[1] < 2:
                st.error(
                    "No hay suficientes datos históricos "
                    "para realizar la optimización."
                )
                st.stop()

            tickers = data.columns.tolist()

            returns = data.pct_change().dropna()

            if returns.empty:
                st.error("No se pudieron calcular los rendimientos.")
                st.stop()

            # =====================================
            # 2. ESTADÍSTICAS
            # =====================================

            mu = returns.mean()
            sigma = returns.std()

            # =====================================
            # 3. PROBLEMA DE OPTIMIZACIÓN
            # =====================================

            qp = QuadraticProgram()

            for t in tickers:
                qp.binary_var(name=t)

            linear = {
                t: float(mu[t] - lambda_risk * sigma[t])
                for t in tickers
            }

            qp.maximize(linear=linear)

            # =====================================
            # 4. QAOA
            # =====================================

            sampler = StatevectorSampler(
                default_shots=int(shots),
                seed=42
            )

            optimizer = COBYLA(
                maxiter=int(maxiter)
            )

            qaoa = QAOA(
                sampler=sampler,
                optimizer=optimizer,
                reps=int(reps)
            )

            eigen_optimizer = MinimumEigenOptimizer(qaoa)

            # Sin argumento penalty
            result = eigen_optimizer.solve(qp)

            x = np.rint(result.x).astype(int)

            # =====================================
            # 5. RESULTADOS
            # =====================================

            output = pd.DataFrame({
                "Activo": tickers,
                "Mantener (1=Sí)": x,
                "Vender (1=Sí)": 1 - x,
                "Retorno esperado": mu.values,
                "Volatilidad": sigma.values
            })

            # =====================================
            # 6. INDICADORES
            # =====================================

            selected = output[
                output["Mantener (1=Sí)"] == 1
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Acciones seleccionadas",
                len(selected)
            )

            col2.metric(
                "Activos analizados",
                len(tickers)
            )

            col3.metric(
                "Valor objetivo",
                f"{result.fval:.6f}"
            )

            st.subheader("Decisión óptima")

            st.dataframe(
                output.style.format({
                    "Retorno esperado": "{:.4%}",
                    "Volatilidad": "{:.4%}"
                }),
                use_container_width=True
            )

            st.subheader("Acciones seleccionadas")

            if selected.empty:
                st.warning("El modelo no seleccionó ninguna acción.")
            else:
                st.dataframe(
                    selected,
                    use_container_width=True
                )

            # =====================================
            # 7. GRÁFICA
            # =====================================

            st.subheader("Decisión de inversión")

            chart_data = output.set_index("Activo")[
                ["Mantener (1=Sí)", "Vender (1=Sí)"]
            ]

            st.bar_chart(chart_data)

            # =====================================
            # 8. DESCARGA
            # =====================================

            csv = output.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Descargar resultados CSV",
                data=csv,
                file_name="resultados_qaoa.csv",
                mime="text/csv"
            )

            st.success("Optimización QAOA completada.")

    except Exception as e:

        st.error(f"Error durante la ejecución: {e}")
