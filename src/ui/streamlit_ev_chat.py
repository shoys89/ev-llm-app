import os
import re
import sys
from typing import Optional, Tuple

import streamlit as st

# Aseguramos que el proyecto raíz esté en sys.path para poder importar `src.*`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipeline.ev_pipeline import EVEnergyPipeline

# Instanciamos la pipeline (carga el modelo de Hugging Face)
pipeline = EVEnergyPipeline()


def _extract_numbers_from_text(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int]]:
    """
    Extrae de forma sencilla algunos campos desde texto libre.

    - Capacidad de batería en kWh -> número seguido de 'kWh'
    - SoC inicio / fin           -> dos porcentajes
    - Duración en horas          -> número seguido de 'h', 'hora', 'horas'
    - Año del vehículo           -> año de 4 dígitos (20xx)
    """
    text_lower = text.lower()

    # Capacidad de batería
    batt_match = re.search(r"(\d+(?:[\.,]\d+)?)\s*kwh", text_lower)
    battery_capacity = float(batt_match.group(1).replace(",", ".")) if batt_match else None

    # SoC inicial y final
    pct_matches = re.findall(r"(\d{1,3})\s*%", text_lower)
    soc_start = soc_end = None
    if len(pct_matches) >= 2:
        soc_start = float(pct_matches[0])
        soc_end = float(pct_matches[1])

    # Duración en horas
    dur_match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(h|hora|horas)", text_lower)
    duration_hours = float(dur_match.group(1).replace(",", ".")) if dur_match else None

    # Año del vehículo (opcional)
    year_match = re.search(r"(20\d{2})", text_lower)
    vehicle_year = int(year_match.group(1)) if year_match else None

    return battery_capacity, soc_start, soc_end, duration_hours, vehicle_year


def main():
    st.set_page_config(page_title="Asistente de Carga de Vehículos Eléctricos", page_icon="🔋")
    st.title("🔋 Asistente de Carga de Vehículos Eléctricos")
    st.write(
        "Describe tu sesión de carga (vehículo, SoC inicial/final, duración, capacidad de batería, año, etc.) "
        "y el asistente estimará la energía cargada usando tu modelo de Hugging Face."
    )

    user_text = st.text_area(
        "Descripción de la sesión de carga",
        placeholder="Ejemplo: Tengo un EV con batería de 75 kWh, lo cargué del 20% al 60% en 1.5 horas, es modelo 2023...",
        height=150,
    )

    if st.button("Calcular energía estimada"):
        if not user_text.strip():
            st.warning("Por favor, ingresa una descripción de la sesión de carga.")
            return

        battery_capacity, soc_start, soc_end, duration_hours, vehicle_year = _extract_numbers_from_text(user_text)

        missing_fields = []
        if battery_capacity is None:
            missing_fields.append("capacidad de batería (kWh)")
        if soc_start is None or soc_end is None:
            missing_fields.append("SoC inicial y final (en %)")
        if duration_hours is None:
            missing_fields.append("duración de la carga (horas)")
        # vehicle_year es opcional

        if missing_fields:
            st.error(
                "Me falta información para poder estimar la energía cargada:\n\n"
                + ", ".join(missing_fields)
                + "\n\nEjemplos:\n"
                "- Capacidad: 'tiene una batería de 60 kWh'\n"
                "- SoC: 'lo cargué del 20% al 80%'\n"
                "- Duración: 'tardó 1.5 horas'\n"
            )
            return

        try:
            pred = pipeline.predict(
                battery_capacity_kwh=battery_capacity,
                soc_start_pct=soc_start,
                soc_end_pct=soc_end,
                charging_duration_hours=duration_hours,
                vehicle_year=vehicle_year,
            )

            st.subheader("✅ Resultados de la sesión")
            st.markdown(f"- **Capacidad de batería**: {battery_capacity:.1f} kWh")
            st.markdown(f"- **SoC inicial**: {soc_start:.1f}%")
            st.markdown(f"- **SoC final**: {soc_end:.1f}%")
            st.markdown(f"- **Diferencia de SoC**: {soc_end - soc_start:.1f} puntos porcentuales")
            st.markdown(f"- **Duración de la carga**: {duration_hours:.2f} horas")
            if vehicle_year is not None:
                st.markdown(f"- **Año del vehículo**: {vehicle_year}")

            st.markdown("---")
            st.markdown(
                f"🔌 **Energía estimada cargada**: **{pred:.2f} kWh** "
                "(según el modelo de predicción)."
            )
            st.info(
                "Esta es una estimación basada en patrones de datos históricos. "
                "En la práctica, pueden existir variaciones dependiendo del cargador, "
                "la temperatura, el estado de la batería y otras condiciones."
            )

        except Exception as exc:
            st.error(
                "Ocurrió un error al intentar calcular la predicción:\n\n"
                f"`{exc}`\n\n"
                "Por favor, verifica los datos o inténtalo de nuevo."
            )


if __name__ == "__main__":
    main()
