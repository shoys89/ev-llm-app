
"""Ejemplo simple por consola para la pipeline de energía EV."""

from src.pipeline.ev_pipeline import EVEnergyPipeline


def main():
    pipeline = EVEnergyPipeline()
    battery_capacity_kwh = 75.0
    soc_start_pct = 20.0
    soc_end_pct = 60.0
    charging_duration_hours = 1.5
    vehicle_year = 2023

    pred = pipeline.predict(
        battery_capacity_kwh=battery_capacity_kwh,
        soc_start_pct=soc_start_pct,
        soc_end_pct=soc_end_pct,
        charging_duration_hours=charging_duration_hours,
        vehicle_year=vehicle_year,
    )

    print(f"Predicción de energía consumida: {pred:.2f} kWh")


if __name__ == "__main__":
    main()
