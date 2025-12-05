
from typing import Optional

from src.model.ev_model import EVEnergyModel, DEFAULT_REPO_ID
from src.core.session_completer import SessionCompleter, SessionInfo


class EVEnergyPipeline:
    """Arquitectura mínima de inferencia.

    Uso:
        pipeline = EVEnergyPipeline()
        pred = pipeline.predict(
            battery_capacity_kwh=75,
            soc_start_pct=20,
            soc_end_pct=60,
            charging_duration_hours=1.5,
            vehicle_year=2023,
        )
    """

    def __init__(self, repo_id: str = None, force_download: bool = False):
        self.session_completer = SessionCompleter()
        self.model = EVEnergyModel(repo_id=repo_id or DEFAULT_REPO_ID,
                                   force_download=force_download)

    def build_session(
        self,
        battery_capacity_kwh: float,
        soc_start_pct: float,
        soc_end_pct: float,
        charging_duration_hours: float,
        vehicle_year: Optional[int] = None,
    ) -> SessionInfo:
        return self.session_completer.build_from_raw(
            battery_capacity_kwh=battery_capacity_kwh,
            soc_start_pct=soc_start_pct,
            soc_end_pct=soc_end_pct,
            charging_duration_hours=charging_duration_hours,
            vehicle_year=vehicle_year,
        )

    def predict(
        self,
        battery_capacity_kwh: float,
        soc_start_pct: float,
        soc_end_pct: float,
        charging_duration_hours: float,
        vehicle_year: Optional[int] = None,
    ) -> float:
        """Devuelve la predicción de energía consumida (kWh)."""
        session = self.build_session(
            battery_capacity_kwh=battery_capacity_kwh,
            soc_start_pct=soc_start_pct,
            soc_end_pct=soc_end_pct,
            charging_duration_hours=charging_duration_hours,
            vehicle_year=vehicle_year,
        )
        preds = self.model.predict_from_session(session.to_model_dict())
        return float(preds[0])
