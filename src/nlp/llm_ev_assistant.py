import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import unicodedata
import pandas as pd

# Aseguramos que el proyecto raíz esté en sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model.ev_model import EVEnergyModel

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()
# ----------------------------------------------------
# Configuración: rutas, modelo HF, LLM, dataset EV-DB
# ----------------------------------------------------

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "EV-DB.csv")  # ajusta si tu CSV está en otro lugar

HF_REPO_ID = "mchacongucenfotec/ev-test-train"

## TODO Remove hardcoded keys for demo purposes only

# Debes tener estas variables en tu entorno:
# export HF_TOKEN="..."
# export GROQ_API_KEY="..."

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Carga de dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró EV-DB.csv en {DATA_PATH}")

df_ev = pd.read_csv(DATA_PATH)

# Modelo de energía en HF
ev_model = EVEnergyModel(repo_id=HF_REPO_ID, force_download=False)

# LLM en Groq
llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=GROQ_API_KEY,
)

parser = JsonOutputParser()

# -------------------------
# 1) Extracción estructurada
# -------------------------

extract_prompt = PromptTemplate(
    template="""
Eres un asistente que extrae información estructurada sobre una sesión de carga de un vehículo eléctrico.

A partir del siguiente mensaje del usuario:

\"\"\"{user_msg}\"\"\"

Debes devolver un JSON EXACTO con esta forma:

{format_instructions}

Campos:
- brand: marca del vehículo (texto) o null si no se menciona.
- model: modelo del vehículo (texto) o null si no se menciona.
- soc_start: SoC inicial en %, número o null.
- soc_end: SoC final en %, número o null.
- duration_hours: duración aproximada de la carga en horas, número (ej. 1.5) o null.
""",
    input_variables=["user_msg"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def extract_session_info(user_msg: str) -> Dict[str, Any]:
    """Llama al LLM para extraer brand, model, soc_start, soc_end, duration_hours."""
    chain = extract_prompt | llm | parser
    extracted = chain.invoke({"user_msg": user_msg})
    # esperado: {"brand": ..., "model": ..., "soc_start": ..., "soc_end": ..., "duration_hours": ...}
    return extracted


# -------------------------
# 2) Búsqueda de vehículo en EV-DB
# -------------------------

def _normalize_text(s: str) -> str:
    """Minimiza diferencias: minúsculas, sin tildes, sin espacios duplicados."""
    s = s.strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = " ".join(s.split())
    return s


def find_vehicle_row(brand: Optional[str], model: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Intenta encontrar una fila en EV-DB usando coincidencias FLEXIBLES
    entre (brand, model) del LLM y las columnas BRAND / MODEL del CSV.
    """
    if not brand or not model:
        print(f"⚠️ find_vehicle_row: brand/model vacíos -> brand={brand!r}, model={model!r}")
        return None

    brand_n = _normalize_text(str(brand))
    model_n = _normalize_text(str(model))

    df_tmp = df_ev.copy()
    df_tmp["BRAND_N"] = df_tmp["BRAND"].astype(str).apply(_normalize_text)
    df_tmp["MODEL_N"] = df_tmp["MODEL"].astype(str).apply(_normalize_text)

    print(f"🔍 Buscando vehículo para brand='{brand_n}', model='{model_n}'")

    # 1) match exacto normalizado
    mask_exact = (df_tmp["BRAND_N"] == brand_n) & (df_tmp["MODEL_N"] == model_n)
    subset = df_tmp[mask_exact]
    if len(subset) > 0:
        print("✅ Match EXACTO encontrado")
        return subset.iloc[0].to_dict()

    # 2) BRAND contiene brand_n y MODEL contiene model_n (subcadena)
    mask_contains = (
        df_tmp["BRAND_N"].str.contains(brand_n, na=False) &
        df_tmp["MODEL_N"].str.contains(model_n, na=False)
    )
    subset = df_tmp[mask_contains]
    if len(subset) > 0:
        print("✅ Match por CONTAINS (brand y model) encontrado")
        return subset.iloc[0].to_dict()

    # 3) BRAND coincide y cualquier token relevante del modelo aparece en MODEL
    model_tokens = [t for t in model_n.split() if len(t) >= 3]
    if model_tokens:
        mask_token = df_tmp["BRAND_N"].str.contains(brand_n, na=False)
        for tok in model_tokens:
            mask_token &= df_tmp["MODEL_N"].str.contains(tok, na=False)
        subset = df_tmp[mask_token]
        if len(subset) > 0:
            print(f"✅ Match por TOKENS del modelo ({model_tokens}) encontrado")
            return subset.iloc[0].to_dict()

    # 4) Último recurso: ignorar brand y matchear solo por MODEL (contiene)
    mask_model_only = df_tmp["MODEL_N"].str.contains(model_tokens[0] if model_tokens else model_n, na=False)
    subset = df_tmp[mask_model_only]
    if len(subset) > 0:
        print("✅ Match SOLO por modelo encontrado (brand ignorado)")
        return subset.iloc[0].to_dict()

    print(f"⚠️ No se encontró vehículo para brand='{brand}' model='{model}'")
    return None

# -------------------------
# 3) Completar sesión + cálculos físicos
# -------------------------

def complete_session_info(
    info: Dict[str, Any],
    vehicle_row: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    info: dict con posibles campos de la sesión
    vehicle_row: fila del dataset EV-DB correspondiente al vehículo seleccionado.
                 Debe contener 'MODEL.1' (año) y 'BATT_CAPACITY' si existen.
    """
    info = info.copy()
    questions: List[str] = []
    from datetime import datetime
    current_year = datetime.now().year

    # 1) Completar con dataset
    if vehicle_row:
        # Battery Capacity desde dataset
        if info.get("Battery Capacity (kWh)") is None and "BATT_CAPACITY" in vehicle_row:
            try:
                info["Battery Capacity (kWh)"] = float(vehicle_row["BATT_CAPACITY"])
            except Exception:
                pass

        # Vehicle Age via year
        if info.get("Vehicle Age (years)") is None and "MODEL.1" in vehicle_row:
            try:
                vehicle_year = int(vehicle_row["MODEL.1"])
                info["Vehicle Age (years)"] = current_year - vehicle_year
            except Exception:
                pass

    # 2) Preguntar por campos críticos
    if info.get("Battery Capacity (kWh)") is None:
        questions.append("¿Cuál es la capacidad de batería de tu vehículo (kWh)?")

    if info.get("SoC_diff") is None:
        questions.append("¿Cuál fue la diferencia de SoC durante la carga (SoC final - SoC inicial, en %)?")

    if info.get("Charging Duration (hours)") is None:
        questions.append("¿Cuánto duró la sesión de carga en horas? (por ejemplo 1.5)")

    if info.get("Vehicle Age (years)") is None:
        questions.append("¿Cuántos años tiene tu vehículo?")

    # 3) Cálculos derivados
    batt = info.get("Battery Capacity (kWh)")
    soc_diff = info.get("SoC_diff")
    duration = info.get("Charging Duration (hours)")

    # Energy_per_SoC
    if batt is not None and info.get("Energy_per_SoC") is None:
        info["Energy_per_SoC"] = batt / 100.0

    # Energy_est_SoC
    if soc_diff is not None and info.get("Energy_per_SoC") is not None:
        if info.get("Energy_est_SoC") is None:
            info["Energy_est_SoC"] = soc_diff * info["Energy_per_SoC"]

    # Charging_Rate
    if info.get("Energy_est_SoC") is not None and duration not in (None, 0):
        if info.get("Charging_Rate") is None:
            info["Charging_Rate"] = info["Energy_est_SoC"] / duration

    # Charge_Efficiency
    if info.get("Charge_Efficiency") is None:
        info["Charge_Efficiency"] = 0.92

    # Power_proxy
    if info.get("Power_proxy") is None and info.get("Charging_Rate") is not None:
        info["Power_proxy"] = info["Charging_Rate"]

    return info, questions


# -------------------------
# 4) Llamar al modelo HF si no faltan datos
# -------------------------

def run_prediction_logic(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquesta:
      - usa brand/model para buscar en EV-DB
      - construye session_info base
      - completa con cálculos físicos
      - si faltan datos: mode = ask_missing
      - si no: mode = predict + llama a modelo HF
    """
    brand = extracted.get("brand")
    model = extracted.get("model")
    soc_start = extracted.get("soc_start")
    soc_end = extracted.get("soc_end")
    duration_hours = extracted.get("duration_hours")

    vehicle_row = find_vehicle_row(brand, model)

    base_session = {
        "Battery Capacity (kWh)": None,
        "SoC_diff": None,
        "Charging Duration (hours)": duration_hours,
        "Energy_est_SoC": None,
        "Charging_Rate": None,
        "Power_proxy": None,
        "Charge_Efficiency": None,
        "Energy_per_SoC": None,
        "Vehicle Age (years)": None,
    }

    if soc_start is not None and soc_end is not None:
        base_session["SoC_diff"] = soc_end - soc_start

    session_info, questions = complete_session_info(base_session, vehicle_row=vehicle_row)

    result: Dict[str, Any] = {
        "extracted": extracted,
        "vehicle_row": vehicle_row,
        "session_info": session_info,
        "questions": questions,
        "mode": None,
        "prediction": None,
    }

    if questions:
        result["mode"] = "ask_missing"
        return result

    # Si no faltan datos, llamamos al modelo HF
    pred_value = ev_model.predict_from_session(session_info)[0]
    result["mode"] = "predict"
    result["prediction"] = float(pred_value)
    return result


# -------------------------
# 5) Prompt final para el LLM
# -------------------------

final_prompt = PromptTemplate.from_template("""
Eres un asistente especializado en vehículos eléctricos.

Datos extraídos del usuario:
{extracted}

Fila de vehículo encontrada (si existe):
{vehicle_row}

Información de la sesión de carga calculada:
{session_info}

Preguntas pendientes (si hay campos faltantes):
{questions}

Modo actual:
{mode}

Predicción del modelo (si existe):
{prediction}

Instrucciones:

- Si "mode" es "ask_missing" o la lista "questions" NO está vacía:
  - No inventes ningún valor ni uses la predicción.
  - Pregunta al usuario esas cosas en español, de forma clara y amable.
  - Puedes mencionar qué información ya tienes y qué te falta.

- Si "mode" es "predict" y la lista "questions" está vacía:
  - Explica al usuario, en español y de forma clara:
    - Qué vehículo has detectado (marca, modelo, año si está disponible).
    - Cuánta energía estimada se cargó (Energy_est_SoC, kWh).
    - La potencia media de carga (Charging_Rate, kW).
    - La eficiencia de carga asumida o calculada (Charge_Efficiency).
    - La edad estimada del vehículo (Vehicle Age, años).
  - Explica también el resultado de la predicción del modelo:
    - Indica el valor de la predicción (en kWh) y qué significa para el usuario.
  - Puedes mostrar el JSON final de forma compacta si es útil, pero explícalo en lenguaje natural.

Responde SOLO en español.
""")


def run_llm_assistant(user_msg: str) -> str:
    """
    Entrada: mensaje libre del usuario.
    Salida: respuesta en español generada por el LLM,
            usando extracción estructurada + modelo HF.
    """
    # 1) Extraer info con el LLM
    extracted = extract_session_info(user_msg)

    # 2) Lógica Python: completar sesión y (posible) predicción
    logic_result = run_prediction_logic(extracted)

    # 3) Preparar campos para el prompt final
    import json
    prompt_input = {
        "extracted": json.dumps(logic_result["extracted"], ensure_ascii=False, indent=2),
        "vehicle_row": json.dumps(logic_result["vehicle_row"], ensure_ascii=False, indent=2) if logic_result["vehicle_row"] else "null",
        "session_info": json.dumps(logic_result["session_info"], ensure_ascii=False, indent=2),
        "questions": json.dumps(logic_result["questions"], ensure_ascii=False, indent=2),
        "mode": logic_result["mode"],
        "prediction": logic_result["prediction"],
    }

    # 4) LLM genera la respuesta final
    prompt = final_prompt.format(**prompt_input)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
