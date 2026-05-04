# APIs Gratuitas para Dados Reais de Sensores

## 🌡️ **OpenWeatherMap** (Recomendado)

### ✅ **Plano Gratuito**
- **1.000.000 chamadas/mês** 
- **60 chamadas/minuto**
- **Dados globais** de temperatura e umidade
- **API Poluição** com CO2 estimado via AQI

### 📊 **Dados Disponíveis**
- **Temperatura**: Atual, umidade, pressão
- **Qualidade do Ar**: AQI, CO, NO2, O3, SO2, PM2.5, PM10
- **Localização**: Latitude/longitude global
- **Histórico**: Dados históricos (plano estudante)

### 🔑 **Como Usar**
```bash
# 1. Criar conta gratuita
https://home.openweathermap.org/users/sign_up

# 2. Obter API Key
https://home.openweathermap.org/api_keys

# 3. Configurar na API
curl -X POST "http://localhost:8001/configurar-api" \
  -H "Content-Type: application/json" \
  -d '"fed50354734dc512fd4011e6abb2fecd"'
```

### 📡 **Endpoints**
```python
# Current Weather
https://api.openweathermap.org/data/2.5/weather?lat=-23.55&lon=-46.63&appid=KEY&units=metric

# Air Pollution  
https://api.openweathermap.org/data/2.5/air_pollution?lat=-23.55&lon=-46.63&appid=KEY
```

---

## 🏭 **Outras APIs Gratuitas**

### **Airly API**
- **Dados reais** de estações na Europa
- **Plano gratuito** limitado
- **Qualidade do ar** em tempo real

### **Adafruit IO**
- **Platforma IoT** gratuita
- **Sensores conectados** de usuários
- **Dashboard** visual
- **Limitado** a dispositivos conectados

### **ThingSpeak**
- **Platforma IoT** para makers
- **Dados públicos** de sensores
- **Gráficos** em tempo real
- **Limitado** a canais públicos

---

## 🚀 **API Integrada (Criada)**

Criei `api_sensores_reais.py` que integra:

### **Funcionalidades**
- ✅ **Dados reais** do OpenWeatherMap
- ✅ **Fallback** automático para simulados
- ✅ **Streaming** em tempo real
- ✅ **Clustering** K-means
- ✅ **Gráficos** e visualizações

### **Endpoints Novos**
- **`POST /configurar-api`** - Configurar API key
- **`POST /treinar`** - Treinar com dados reais
- **`GET /dados/atual`** - Leitura real atual
- **`GET /dados/stream-sse`** - Streaming real

---

## 📈 **Comparativo**

| API | Dados Reais | Gratis | Limites | Temperatura | Umidade | CO2 |
|-----|-------------|---------|----------|-------------|---------|-----|
| **OpenWeatherMap** | ✅ | ✅ | 1M/mês | ✅ | ✅ | 🔄* |
| **Airly** | ✅ | ✅ | Limitado | ❌ | ❌ | ✅ |
| **Adafruit IO** | ✅ | ✅ | Dispositivos | ✅ | ✅ | ❌ |
| **ThingSpeak** | ✅ | ✅ | Públicos | ✅ | ✅ | ❌ |

*CO2 estimado via AQI (Air Quality Index)

---

## 🎯 **Recomendação**

**Use OpenWeatherMap** porque:
- ✅ **Mais completa** (temperatura + umidade + qualidade do ar)
- ✅ **Global** (qualquer localização)
- ✅ **Estável** (API profissional)
- ✅ **Limites generosos** (1M chamadas/mês)
- ✅ **Documentação** excelente

---

## 🔧 **Exemplo de Uso**

### **1. Configurar API**
```python
import requests

# Configurar API key
response = requests.post(
    "http://localhost:8001/configurar-api",
    json="fed50354734dc512fd4011e6abb2fecd"
)
print(response.json())
```

### **2. Treinar Modelo**
```python
# Treinar com dados reais
response = requests.post(
    "http://localhost:8001/treinar?use_real_data=true&n_dias_historico=7"
)
print(response.json())
```

### **3. Obter Dados em Tempo Real**
```python
# Leitura atual real
response = requests.get("http://localhost:8001/dados/atual")
print(response.json())

# Streaming em tempo real
import sseclient
messages = sseclient.SSEClient(
    "http://localhost:8001/dados/stream-sse"
)
for msg in messages:
    print(msg.data)
```

---

## 📊 **Formato dos Dados**

```json
{
  "timestamp": "2026-04-26T21:30:00.000000",
  "temperatura": 23.5,
  "umidade": 65.2,
  "co2": 450,
  "source": "OpenWeatherMap",
  "location": "São Paulo",
  "aqi": 1,
  "cluster": 0,
  "nome_grupo": "Normal"
}
```

---

## 🌍 **Localizações Suportadas**

Altere as coordenadas em `LOCATION`:
```python
LOCATION = {
    "lat": -23.5505,  # São Paulo
    "lon": -46.6333,
    "city": "São Paulo"
}
```

**Exemplos:**
- **NYC**: 40.7128, -74.0060
- **Londres**: 51.5074, -0.1278
- **Tóquio**: 35.6762, 139.6503

---

## 🚨 **Limitações Importantes**

### **OpenWeatherMap**
- **CO2 não é direto** (estimado via AQI)
- **Rate limits**: 60 chamadas/minuto
- **Precisão**: Baseada em modelos meteorológicos

### **Solução**
- **Combinar dados reais** com simulação
- **Cache** para respeitar rate limits
- **Fallback** automático se API falhar

---

## 📱 **Dashboard Web**

A API integrada inclui:
- **Visualização** em tempo real
- **Gráficos** de dispersão
- **Distribuição** dos clusters
- **Streaming** via Server-Sent Events

Acesse: **http://localhost:8001**
