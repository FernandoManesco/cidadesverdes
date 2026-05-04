from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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
import base64
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict
import asyncio
from threading import Thread
import json
import requests
import os

app = FastAPI(title="API K-means Sensores Reais", version="3.0.0")

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
    "dados_historicos": [],
    "ultima_atualizacao": None,
    "api_key": None,
}

CORES = ["#00E5FF", "#FF4C6A", "#69FF47", "#FFD700"]
NOMES_GRUPOS = {
    0: "Normal",
    1: "Quente + CO₂ Alto",
    2: "Frio + Úmido", 
    3: "CO₂ Crítico",
}

# Configurações APIs
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "fed50354734dc512fd4011e6abb2fecd")
LOCATION = {
    "lat": -22.4114,  # Rio Claro (22º 24' 41" S)
    "lon": -47.5614,  # Rio Claro (47º 33' 41" W)
    "city": "Rio Claro"
}

class SensoresReaisAPI:
    def __init__(self):
        self.openweather_key = OPENWEATHER_KEY
        
    def get_openweather_data(self):
        """Obtém dados reais do OpenWeatherMap API"""
        if not self.openweather_key:
            raise HTTPException(status_code=401, detail="OpenWeather API key não configurada")
        
        try:
            # Current Weather
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LOCATION['lat']}&lon={LOCATION['lon']}&appid={self.openweather_key}&units=metric"
            weather_response = requests.get(weather_url)
            weather_data = weather_response.json()
            
            # Air Pollution
            pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LOCATION['lat']}&lon={LOCATION['lon']}&appid={self.openweather_key}"
            pollution_response = requests.get(pollution_url)
            pollution_data = pollution_response.json()
            
            if weather_response.status_code != 200 or pollution_response.status_code != 200:
                raise Exception("Erro na API")
            
            # Extrair dados
            main = weather_data.get('main', {})
            temp = main.get('temp', 20)
            humidity = main.get('humidity', 50)
            
            # CO2 estimado baseado em AQI (Air Quality Index)
            aqi = pollution_data.get('list', [{}])[0].get('main', {}).get('aqi', 1)
            
            # Converter AQI para CO2 estimado (aproximação)
            co2_mapping = {1: 400, 2: 600, 3: 800, 4: 1200, 5: 2000}
            co2 = co2_mapping.get(aqi, 400)
            
            # Adicionar variação realista
            temp += random.uniform(-2, 2)
            humidity += random.uniform(-5, 5)
            co2 += random.uniform(-50, 50)
            
            return {
                "timestamp": datetime.now(),
                "temperatura": round(temp, 1),
                "umidade": round(max(20, min(95, humidity)), 1),
                "co2": int(max(300, min(5000, co2))),
                "source": "OpenWeatherMap",
                "location": LOCATION['city'],
                "aqi": aqi
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao obter dados: {str(e)}")
    
    def get_historical_data(self, days: int = 7):
        """Gera dados históricos simulados baseados em padrões reais"""
        leituras = []
        base_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 24):  # Uma leitura por hora
            timestamp = base_time + timedelta(hours=i)
            hora = timestamp.hour
            
            # Simular padrões diários
            if 6 <= hora <= 18:  # Dia
                temp_base = 22 + random.uniform(-3, 5)
                humidity_base = 45 + random.uniform(-10, 15)
                co2_base = 450 + random.uniform(-100, 200)
            else:  # Noite
                temp_base = 18 + random.uniform(-3, 3)
                humidity_base = 60 + random.uniform(-10, 20)
                co2_base = 600 + random.uniform(-100, 300)
            
            # Adicionar anomalias ocasionais
            if random.random() < 0.1:
                anomalia = random.choice(['quente', 'frio', 'co2_alto'])
                if anomalia == 'quente':
                    temp_base = random.uniform(30, 35)
                    co2_base = random.uniform(800, 1500)
                elif anomalia == 'frio':
                    temp_base = random.uniform(8, 12)
                    humidity_base = random.uniform(75, 85)
                elif anomalia == 'co2_alto':
                    co2_base = random.uniform(2500, 4000)
            
            leituras.append({
                "timestamp": timestamp,
                "temperatura": round(temp_base, 1),
                "umidade": round(humidity_base, 1),
                "co2": int(co2_base),
                "source": "Simulated Historical"
            })
        
        return leituras
    
    def get_nearby_cities_data(self):
        """Obtém dados das 10 cidades mais próximas de Rio Claro"""
        # Coordenadas das 10 cidades mais próximas de Rio Claro/SP
        cities = [
            {"name": "Piracicaba", "lat": -22.7253, "lon": -47.6492},
            {"name": "Limeira", "lat": -22.5648, "lon": -47.4017},
            {"name": "Araras", "lat": -22.3544, "lon": -47.3847},
            {"name": "Cordeirópolis", "lat": -22.4833, "lon": -47.4583},
            {"name": "Santa Gertrudes", "lat": -22.4689, "lon": -47.5333},
            {"name": "Ipeúna", "lat": -22.4167, "lon": -47.6833},
            {"name": "Corumbataí", "lat": -22.2167, "lon": -47.6167},
            {"name": "Itirapina", "lat": -22.2500, "lon": -47.7833},
            {"name": "Rio das Pedras", "lat": -22.5000, "lon": -47.5333},
            {"name": "Charqueada", "lat": -22.5000, "lon": -47.7667}
        ]
        
        dados_cidades = []
        
        for city in cities:
            try:
                # Current Weather
                weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={city['lat']}&lon={city['lon']}&appid={self.openweather_key}&units=metric"
                weather_response = requests.get(weather_url)
                weather_data = weather_response.json()
                
                # Air Pollution
                pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={city['lat']}&lon={city['lon']}&appid={self.openweather_key}"
                pollution_response = requests.get(pollution_url)
                pollution_data = pollution_response.json()
                
                if weather_response.status_code == 200 and pollution_response.status_code == 200:
                    # Extrair dados
                    main = weather_data.get('main', {})
                    temp = main.get('temp', 20)
                    humidity = main.get('humidity', 50)
                    
                    # Obter dados completos de poluição
                    pollution_info = pollution_data.get('list', [{}])[0]
                    aqi = pollution_info.get('main', {}).get('aqi', 1)
                    components = pollution_info.get('components', {})
                    
                    # Calcular AQI mais preciso baseado em múltiplos poluentes
                    pm25 = components.get('pm2_5', 0)
                    pm10 = components.get('pm10', 0)
                    no2 = components.get('no2', 0)
                    o3 = components.get('o3', 0)
                    co = components.get('co', 0)
                    
                    # Estimar CO2 baseado nos componentes de poluição
                    # Fórmula melhorada: baseada em CO e outros poluentes
                    co2_base = 400  # CO2 atmosférico base
                    
                    # Adicionar contribuição do CO (monóxido de carbono)
                    if co > 100:
                        co2_base += (co - 100) * 2  # Cada 100µg/m³ de CO ≈ 200ppm de CO2
                    
                    # Adicionar contribuição de NO2 (indicador de tráfego/indústria)
                    if no2 > 20:
                        co2_base += (no2 - 20) * 5  # Cada 20µg/m³ de NO2 ≈ 100ppm de CO2
                    
                    # Adicionar contribuição de PM2.5 (partículas finas)
                    if pm25 > 10:
                        co2_base += (pm25 - 10) * 8  # Cada 10µg/m³ de PM2.5 ≈ 80ppm de CO2
                    
                    # Ajuste baseado no AQI geral
                    if aqi >= 4:  # Poor/Very Poor
                        co2_base *= 1.5
                    elif aqi == 3:  # Moderate
                        co2_base *= 1.2
                    
                    co2 = co2_base
                    
                    # Adicionar variação realista
                    temp += random.uniform(-2, 2)
                    humidity += random.uniform(-5, 5)
                    co2 += random.uniform(-50, 50)
                    
                    dados_cidades.append({
                        "nome": city['name'],
                        "temperatura": round(temp, 1),
                        "umidade": round(max(20, min(95, humidity)), 1),
                        "co2": int(max(300, min(5000, co2))),
                        "source": "OpenWeatherMap",
                        "location": city['name'],
                        "aqi": aqi,
                        "lat": city['lat'],
                        "lon": city['lon'],
                        "poluents": {
                            "pm2_5": pm25,
                            "pm10": pm10,
                            "no2": no2,
                            "o3": o3,
                            "co": co,
                            "so2": components.get('so2', 0),
                            "nh3": components.get('nh3', 0)
                        }
                    })
                else:
                    # Fallback com dados simulados se API falhar
                    temp = 20 + random.uniform(-5, 10)
                    humidity = 60 + random.uniform(-20, 20)
                    co2 = 500 + random.uniform(-200, 500)
                    
                    dados_cidades.append({
                        "nome": city['name'],
                        "temperatura": round(temp, 1),
                        "umidade": round(max(20, min(95, humidity)), 1),
                        "co2": int(max(300, min(5000, co2))),
                        "source": "Simulated Fallback",
                        "location": city['name'],
                        "aqi": 1,
                        "lat": city['lat'],
                        "lon": city['lon']
                    })
                    
            except Exception as e:
                # Dados simulados em caso de erro
                temp = 20 + random.uniform(-5, 10)
                humidity = 60 + random.uniform(-20, 20)
                co2 = 500 + random.uniform(-200, 500)
                
                dados_cidades.append({
                    "nome": city['name'],
                    "temperatura": round(temp, 1),
                    "umidade": round(max(20, min(95, humidity)), 1),
                    "co2": int(max(300, min(5000, co2))),
                    "source": "Simulated Error",
                    "location": city['name'],
                    "aqi": 1,
                    "lat": city['lat'],
                    "lon": city['lon']
                })
        
        return dados_cidades

sensores_api = SensoresReaisAPI()

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

def classificar_por_aqi(cidade):
    """Classifica cidade baseado no AQI da OpenWeatherMap (1-5)"""
    aqi = cidade.get('aqi', 1)
    
    if aqi == 1:
        return {
            "grupo": "Excelente",
            "nivel": "AQI 1 - Excelente",
            "descricao": "Qualidade do ar excelente",
            "cor": "green",
            "aqi": aqi
        }
    elif aqi == 2:
        return {
            "grupo": "Bom", 
            "nivel": "AQI 2 - Bom",
            "descricao": "Qualidade do ar boa",
            "cor": "lightgreen",
            "aqi": aqi
        }
    elif aqi == 3:
        return {
            "grupo": "Moderado",
            "nivel": "AQI 3 - Moderado", 
            "descricao": "Qualidade do ar moderada",
            "cor": "yellow",
            "aqi": aqi
        }
    elif aqi == 4:
        return {
            "grupo": "Ruim",
            "nivel": "AQI 4 - Ruim",
            "descricao": "Qualidade do ar ruim",
            "cor": "orange",
            "aqi": aqi
        }
    elif aqi == 5:
        return {
            "grupo": "Perigoso",
            "nivel": "AQI 5 - Perigoso",
            "descricao": "Qualidade do ar perigosa",
            "cor": "red",
            "aqi": aqi
        }
    else:
        return {
            "grupo": "Excelente",
            "nivel": "AQI 1 - Excelente",
            "descricao": "Qualidade do ar excelente", 
            "cor": "green",
            "aqi": 1
        }

@app.get("/")
def root():
    return {
        "mensagem": "API K-means Sensores Reais funcionando!",
        "api_key_configurada": bool(OPENWEATHER_KEY),
        "location": LOCATION,
        "endpoints": [
            "/configurar-api",
            "/treinar", 
            "/clusters", 
            "/grafico/2d", 
            "/grafico/distribuicao", 
            "/status",
            "/dados/atual",
            "/dados/historico",
            "/dados/cidades-proximas",
            "/dados/stream-sse"
        ]
    }

@app.post("/configurar-api")
def configurar_api(api_key: str):
    """Configura a chave da API OpenWeatherMap"""
    global OPENWEATHER_KEY
    OPENWEATHER_KEY = api_key
    sensores_api.openweather_key = api_key
    modelo_estado["api_key"] = api_key
    
    return {"status": "API key configurada com sucesso!", "servico": "OpenWeatherMap"}

@app.post("/treinar")
async def treinar(use_real_data: bool = True, n_dias_historico: int = 7, n_clusters: int = 4):
    """Treina o modelo com dados reais ou simulados."""
    
    if use_real_data and not OPENWEATHER_KEY:
        raise HTTPException(status_code=400, detail="Configure API key primeiro com POST /configurar-api")
    
    # Obter dados
    if use_real_data:
        try:
            # Obter leitura atual real
            leitura_atual = sensores_api.get_openweather_data()
            
            # Gerar dados históricos simulados (baseados em padrões reais)
            leituras = sensores_api.get_historical_data(n_dias_historico)
            leituras.append(leitura_atual)  # Adicionar leitura atual
            
        except Exception as e:
            # Fallback para dados simulados se API falhar
            leituras = sensores_api.get_historical_data(n_dias_historico)
    else:
        leituras = sensores_api.get_historical_data(n_dias_historico)
    
    # Converter para DataFrame
    df = pd.DataFrame(leituras)
    df = df.drop(['timestamp', 'source'], axis=1, errors='ignore')
    
    X = df[["temperatura", "umidade", "co2"]].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Atualizar estado
    modelo_estado["kmeans"] = kmeans
    modelo_estado["scaler"] = scaler
    modelo_estado["df"] = df
    modelo_estado["labels"] = labels
    modelo_estado["n_clusters"] = n_clusters
    modelo_estado["dados_historicos"] = leituras
    modelo_estado["ultima_atualizacao"] = datetime.now()
    
    # Calcular informações dos grupos
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
        "fonte_dados": "API Real" if use_real_data else "Simulado",
        "total_amostras": len(df),
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 4),
        "grupos": grupos_info,
        "data_geracao": datetime.now().isoformat(),
        "periodo_dados": {
            "inicio": leituras[0]["timestamp"].isoformat(),
            "fim": leituras[-1]["timestamp"].isoformat()
        }
    }

@app.get("/dados/atual")
def dados_atual():
    """Retorna leitura atual dos sensores (API real ou simulado)."""
    
    if OPENWEATHER_KEY:
        try:
            leitura = sensores_api.get_openweather_data()
        except:
            # Fallback para simulado se API falhar
            leitura = sensores_api.get_historical_data(1)[0]
    else:
        leitura = sensores_api.get_historical_data(1)[0]
    
    # Classificar se modelo treinado
    if modelo_estado["kmeans"] is not None:
        scaler = modelo_estado["scaler"]
        kmeans = modelo_estado["kmeans"]
        centroides = scaler.inverse_transform(kmeans.cluster_centers_)
        
        X = np.array([[leitura["temperatura"], leitura["umidade"], leitura["co2"]]])
        X_scaled = scaler.transform(X)
        cluster = kmeans.predict(X_scaled)[0]
        
        leitura["cluster"] = int(cluster)
        leitura["nome_grupo"] = detectar_nome_grupo(centroides[cluster])
    
    return leitura

@app.get("/dados/historico")
def dados_historico(ultimas_n: int = 50):
    """Retorna as últimas N leituras históricas."""
    if not modelo_estado["dados_historicos"]:
        raise HTTPException(status_code=400, detail="Nenhum dado histórico. Use POST /treinar primeiro.")
    
    historico = modelo_estado["dados_historicos"][-ultimas_n:]
    
    # Adicionar classificações se modelo treinado
    if modelo_estado["kmeans"] is not None:
        scaler = modelo_estado["scaler"]
        kmeans = modelo_estado["kmeans"]
        centroides = scaler.inverse_transform(kmeans.cluster_centers_)
        
        for leitura in historico:
            X = np.array([[leitura["temperatura"], leitura["umidade"], leitura["co2"]]])
            X_scaled = scaler.transform(X)
            cluster = kmeans.predict(X_scaled)[0]
            leitura["cluster"] = int(cluster)
            leitura["nome_grupo"] = detectar_nome_grupo(centroides[cluster])
    
    return {
        "total": len(historico),
        "dados": historico,
        "ultima_atualizacao": modelo_estado["ultima_atualizacao"].isoformat() if modelo_estado["ultima_atualizacao"] else None
    }

@app.get("/dados/cidades-proximas")
def dados_cidades_proximas():
    """Retorna dados das 10 cidades mais próximas de Rio Claro classificadas por AQI."""
    try:
        cidades = sensores_api.get_nearby_cities_data()
        
        # Classificar por AQI (novo sistema)
        for cidade in cidades:
            classificacao_aqi = classificar_por_aqi(cidade)
            cidade["classificacao_aqi"] = classificacao_aqi
            cidade["grupo"] = classificacao_aqi["grupo"]
            cidade["nivel"] = classificacao_aqi["nivel"]
            cidade["descricao"] = classificacao_aqi["descricao"]
            cidade["cor"] = classificacao_aqi["cor"]
        
        # Manter classificação K-means para compatibilidade (futura página de bairros)
        if modelo_estado["kmeans"] is not None:
            scaler = modelo_estado["scaler"]
            kmeans = modelo_estado["kmeans"]
            centroides = scaler.inverse_transform(kmeans.cluster_centers_)
            
            for cidade in cidades:
                X = np.array([[cidade["temperatura"], cidade["umidade"], cidade["co2"]]])
                X_scaled = scaler.transform(X)
                cluster = kmeans.predict(X_scaled)[0]
                cidade["cluster_kmeans"] = int(cluster)
                cidade["nome_grupo_kmeans"] = detectar_nome_grupo(centroides[cluster])
        
        return {
            "total": len(cidades),
            "dados": cidades,
            "referencia": "Rio Claro, SP",
            "sistema_classificacao": "AQI OpenWeatherMap (1-5)",
            "data_geracao": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter dados das cidades: {str(e)}")

@app.get("/dados/stream-sse")
async def dados_stream_sse():
    """Server-Sent Events para streaming em tempo real."""
    
    async def generate():
        while True:
            if OPENWEATHER_KEY:
                try:
                    leitura = sensores_api.get_openweather_data()
                except:
                    leitura = sensores_api.get_historical_data(1)[0]
            else:
                leitura = sensores_api.get_historical_data(1)[0]
            
            # Classificar se modelo disponível
            if modelo_estado["kmeans"] is not None:
                scaler = modelo_estado["scaler"]
                kmeans = modelo_estado["kmeans"]
                centroides = scaler.inverse_transform(kmeans.cluster_centers_)
                
                X = np.array([[leitura["temperatura"], leitura["umidade"], leitura["co2"]]])
                X_scaled = scaler.transform(X)
                cluster = kmeans.predict(X_scaled)[0]
                leitura["cluster"] = int(cluster)
                leitura["nome_grupo"] = detectar_nome_grupo(centroides[cluster])
            
            yield f"data: {json.dumps(leitura)}\n\n"
            await asyncio.sleep(10)  # Nova leitura a cada 10 segundos (respeitando limites da API)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/status")
def status():
    if modelo_estado["kmeans"] is None:
        return {
            "treinado": False, 
            "mensagem": "Modelo ainda não treinado. Use POST /treinar",
            "api_key_configurada": bool(OPENWEATHER_KEY),
            "location": LOCATION
        }
    
    labels = modelo_estado["labels"]
    n = modelo_estado["n_clusters"]
    
    return {
        "treinado": True,
        "n_clusters": n,
        "total_amostras": len(labels),
        "distribuicao": {str(i): int(np.sum(labels == i)) for i in range(n)},
        "ultima_atualizacao": modelo_estado["ultima_atualizacao"].isoformat(),
        "dados_historicos": len(modelo_estado["dados_historicos"]),
        "api_key_configurada": bool(OPENWEATHER_KEY),
        "location": LOCATION,
        "proxima_atualizacao": (datetime.now() + timedelta(seconds=10)).isoformat()
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
            "timestamp": modelo_estado["dados_historicos"][i]["timestamp"].isoformat() if i < len(modelo_estado["dados_historicos"]) else None,
            "source": modelo_estado["dados_historicos"][i].get("source", "Unknown") if i < len(modelo_estado["dados_historicos"]) else None
        })
    
    return {"total": len(resultado), "dados": resultado[:100]}

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

    fig.suptitle("K-means — Dados Reais de Sensores (2D)", color="white", fontsize=16, fontweight="bold", y=1.02)
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

    fig.suptitle("Distribuição dos Clusters — Dados Reais", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)
    return JSONResponse({"imagem_base64": img_b64, "formato": "png"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
