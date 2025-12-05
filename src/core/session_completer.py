
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class SessionInfo:
    """Información que se envía al modelo de predicción."""

    battery_capacity_kwh: Optional[float] = None
    soc_diff: Optional[float] = None
    charging_duration_hours: Optional[float] = None
    energy_est_soc: Optional[float] = None
    charging_rate: Optional[float] = None
    power_proxy: Optional[float] = None
    charge_efficiency: Optional[float] = None
    energy_per_soc: Optional[float] = None
    vehicle_age_years: Optional[float] = None

    def to_model_dict(self) -> Dict[str, Any]:
        """Diccionario en el formato esperado por el modelo de Hugging Face."""
        return {
            "Battery Capacity (kWh)": self.battery_capacity_kwh,
            "SoC_diff": self.soc_diff,
            "Charging Duration (hours)": self.charging_duration_hours,
            "Energy_est_SoC": self.energy_est_soc,
            "Charging_Rate": self.charging_rate,
            "Power_proxy": self.power_proxy,
            "Charge_Efficiency": self.charge_efficiency,
            "Energy_per_SoC": self.energy_per_soc,
            "Vehicle Age (years)": self.vehicle_age_years,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SessionCompleter:
    """Calcula atributos derivados a partir de entradas crudas."""

    def __init__(self, current_year: Optional[int] = None):
        self.current_year = current_year or datetime.now().year

    def build_from_raw(
        self,
        battery_capacity_kwh: float,
        soc_start_pct: float,
        soc_end_pct: float,
        charging_duration_hours: float,
        vehicle_year: Optional[int] = None,
        default_efficiency: float = 0.92,
    ) -> SessionInfo:
        """Construye un SessionInfo a partir de datos básicos de la sesión de carga."""

        soc_diff = soc_end_pct - soc_start_pct

        vehicle_age_years: Optional[float] = None
        if vehicle_year is not None:
            vehicle_age_years = float(self.current_year - int(vehicle_year))

        energy_per_soc = battery_capacity_kwh / 100.0
        energy_est_soc = soc_diff * energy_per_soc

        charging_rate = None
        power_proxy = None
        if charging_duration_hours and charging_duration_hours > 0:
            charging_rate = energy_est_soc / charging_duration_hours
            power_proxy = charging_rate

        session = SessionInfo(
            battery_capacity_kwh=battery_capacity_kwh,
            soc_diff=soc_diff,
            charging_duration_hours=charging_duration_hours,
            energy_est_soc=energy_est_soc,
            charging_rate=charging_rate,
            power_proxy=power_proxy,
            charge_efficiency=default_efficiency,
            energy_per_soc=energy_per_soc,
            vehicle_age_years=vehicle_age_years,
        )

        return session
