from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
import base64

app = FastAPI(title="API K-means Sensores", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global do modelo
modelo_estado = {
    "kmeans": None,
    "scaler": None,
    "df": None,
    "labels": None,
    "n_clusters": 4,
}

CORES = ["#00E5FF", "#FF4C6A", "#69FF47", "#FFD700"]
NOMES_GRUPOS = {
    0: "Normal",
    1: "Quente + CO₂ Alto",
    2: "Frio + Úmido",
    3: "CO₂ Crítico",
}


def detectar_nome_grupo(centroide):
    """Detecta automaticamente o nome do grupo pelo centróide."""
    temp, umid, co2 = centroide
    if co2 > 2500:
        return "CO₂ Crítico"
    elif temp > 30 and co2 > 1200:
        return "Quente + CO₂ Alto"
    elif temp < 12 and umid > 75:
        return "Frio + Úmido"
    else:
        return "Normal"


@app.get("/")
def root():
    return {"mensagem": "API K-means Sensores funcionando!", "endpoints": ["/treinar", "/clusters", "/grafico/2d", "/grafico/3d", "/grafico/distribuicao", "/status"]}


@app.post("/treinar")
async def treinar(file: UploadFile = File(...), n_clusters: int = 4):
    """Recebe o arquivo TXT e treina o K-means."""
    conteudo = await file.read()
    try:
        df = pd.read_csv(io.StringIO(conteudo.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {str(e)}")

    colunas_esperadas = {"temperatura", "umidade", "co2"}
    if not colunas_esperadas.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Colunas esperadas: temperatura, umidade, co2. Recebidas: {list(df.columns)}")

    X = df[["temperatura", "umidade", "co2"]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    modelo_estado["kmeans"] = kmeans
    modelo_estado["scaler"] = scaler
    modelo_estado["df"] = df
    modelo_estado["labels"] = labels
    modelo_estado["n_clusters"] = n_clusters

    centroides_originais = scaler.inverse_transform(kmeans.cluster_centers_)
    grupos_info = {}
    for i, c in enumerate(centroides_originais):
        nome = detectar_nome_grupo(c)
        grupos_info[str(i)] = {
            "nome": nome,
            "centroide": {
                "temperatura": round(float(c[0]), 2),
                "umidade": round(float(c[1]), 2),
                "co2": int(c[2]),
            },
            "total_pontos": int(np.sum(labels == i)),
        }

    return {
        "status": "Modelo treinado com sucesso!",
        "total_amostras": len(df),
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 4),
        "grupos": grupos_info,
    }


@app.get("/status")
def status():
    if modelo_estado["kmeans"] is None:
        return {"treinado": False, "mensagem": "Modelo ainda não treinado. Use POST /treinar"}
    labels = modelo_estado["labels"]
    n = modelo_estado["n_clusters"]
    return {
        "treinado": True,
        "n_clusters": n,
        "total_amostras": len(labels),
        "distribuicao": {str(i): int(np.sum(labels == i)) for i in range(n)},
    }


@app.get("/clusters")
def clusters():
    if modelo_estado["kmeans"] is None:
        raise HTTPException(status_code=400, detail="Modelo não treinado. Use POST /treinar primeiro.")
    df = modelo_estado["df"].copy()
    df["cluster"] = modelo_estado["labels"]
    scaler = modelo_estado["scaler"]
    centroides = scaler.inverse_transform(modelo_estado["kmeans"].cluster_centers_)
    resultado = []
    for i, row in df.iterrows():
        resultado.append({
            "id": int(i),
            "temperatura": float(row["temperatura"]),
            "umidade": float(row["umidade"]),
            "co2": int(row["co2"]),
            "cluster": int(row["cluster"]),
            "nome_grupo": detectar_nome_grupo(centroides[int(row["cluster"])]),
        })
    return {"total": len(resultado), "dados": resultado[:100]}  # retorna primeiros 100


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/grafico/2d")
def grafico_2d():
    if modelo_estado["kmeans"] is None:
        raise HTTPException(status_code=400, detail="Modelo não treinado.")

    df = modelo_estado["df"].copy()
    labels = modelo_estado["labels"]
    scaler = modelo_estado["scaler"]
    centroides = scaler.inverse_transform(modelo_estado["kmeans"].cluster_centers_)
    n = modelo_estado["n_clusters"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0D1117")

    pares = [
        ("temperatura", "umidade", 0, 1),
        ("temperatura", "co2", 0, 2),
        ("umidade", "co2", 1, 2),
    ]

    for ax, (xlabel, ylabel, xi, yi) in zip(axes, pares):
        ax.set_facecolor("#161B22")
        for cluster_id in range(n):
            mask = labels == cluster_id
            cor = CORES[cluster_id % len(CORES)]
            nome = detectar_nome_grupo(centroides[cluster_id])
            ax.scatter(
                df[xlabel][mask], df[ylabel][mask],
                c=cor, alpha=0.5, s=20, label=nome
            )
            ax.scatter(
                centroides[cluster_id][xi], centroides[cluster_id][yi],
                c=cor, s=200, marker="X", edgecolors="white", linewidths=1.5, zorder=5
            )
        ax.set_xlabel(xlabel.capitalize(), color="white", fontsize=11)
        ax.set_ylabel(ylabel.upper() if ylabel == "co2" else ylabel.capitalize(), color="white", fontsize=11)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.grid(True, color="#21262D", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8, facecolor="#21262D", labelcolor="white", edgecolor="#30363D")

    fig.suptitle("K-means — Agrupamento de Anomalias (2D)", color="white", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)
    return JSONResponse({"imagem_base64": img_b64, "formato": "png"})


@app.get("/grafico/distribuicao")
def grafico_distribuicao():
    if modelo_estado["kmeans"] is None:
        raise HTTPException(status_code=400, detail="Modelo não treinado.")

    labels = modelo_estado["labels"]
    scaler = modelo_estado["scaler"]
    centroides = scaler.inverse_transform(modelo_estado["kmeans"].cluster_centers_)
    n = modelo_estado["n_clusters"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    contagens = [int(np.sum(labels == i)) for i in range(n)]
    nomes = [detectar_nome_grupo(centroides[i]) for i in range(n)]
    cores = [CORES[i % len(CORES)] for i in range(n)]

    # Barras
    ax1.set_facecolor("#161B22")
    bars = ax1.bar(nomes, contagens, color=cores, edgecolor="#0D1117", linewidth=1.5)
    for bar, val in zip(bars, contagens):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
    ax1.set_title("Quantidade por Grupo", color="white", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="white", axis="y")
    ax1.tick_params(colors="white", axis="x", labelrotation=15)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363D")
    ax1.set_facecolor("#161B22")
    ax1.grid(True, axis="y", color="#21262D", linestyle="--", linewidth=0.5)

    # Pizza
    ax2.set_facecolor("#161B22")
    wedges, texts, autotexts = ax2.pie(
        contagens, labels=nomes, colors=cores,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 2}
    )
    for text in texts:
        text.set_color("white")
        text.set_fontsize(9)
    for at in autotexts:
        at.set_color("#0D1117")
        at.set_fontweight("bold")
    ax2.set_title("Proporção dos Grupos", color="white", fontsize=13, fontweight="bold")

    fig.suptitle("Distribuição dos Clusters — K-means", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)
    return JSONResponse({"imagem_base64": img_b64, "formato": "png"})