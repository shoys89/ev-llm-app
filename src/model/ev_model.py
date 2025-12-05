
import sys
from typing import Dict, Any, List, Optional

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "mchacongucenfotec/ev-test-train"


class EVEnergyModel:
    """Wrapper del modelo de energía para EV publicado en Hugging Face.

    Asume que el snapshot contiene un módulo `inference.py` con:
        - predict(session_info) -> list[float] o np.ndarray
        - get_feature_names()   -> iterable de nombres de features
    """

    def __init__(self, repo_id: str = DEFAULT_REPO_ID, force_download: bool = False):
        self.repo_id = repo_id
        self.local_dir: Optional[str] = None
        self.hf_predict = None
        self.feature_names: Optional[list[str]] = None

        self._load_snapshot(force_download=force_download)

    def _load_snapshot(self, force_download: bool = False) -> None:
        """Descarga (o usa caché) del snapshot y carga inference.predict."""
        import os
        
        self.local_dir = snapshot_download(repo_id=self.repo_id, force_download=force_download)

        if self.local_dir not in sys.path:
            sys.path.insert(0, self.local_dir)

        # Change to snapshot directory so relative paths in inference.py work correctly
        original_dir = os.getcwd()
        os.chdir(self.local_dir)
        
        try:
            from inference import predict as hf_predict, get_feature_names  # type: ignore
        except Exception as exc:
            os.chdir(original_dir)
            raise RuntimeError(
                f"No se pudo importar 'inference.predict' desde el snapshot en {self.local_dir}. "
                f"Error: {str(exc)}"
            ) from exc

        os.chdir(original_dir)
        
        self.hf_predict = hf_predict
        try:
            self.feature_names = list(get_feature_names())
        except Exception:
            self.feature_names = None

    def predict_from_session(self, session_info: Dict[str, Any]) -> List[float]:
        """Llama a inference.predict(session_info) y devuelve una lista de floats."""
        if self.hf_predict is None:
            raise RuntimeError("El modelo aún no ha sido cargado correctamente.")

        preds = self.hf_predict(session_info)
        if isinstance(preds, (list, tuple)):
            return [float(p) for p in preds]
        try:
            # soporta arrays tipo numpy
            return [float(p) for p in preds]  # type: ignore
        except Exception:
            return [float(preds)]
