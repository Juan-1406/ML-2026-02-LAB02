"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros
from src.excepciones import EtapaPendienteAlumno
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    TODO(alumno): implementar ``cargar`` y devolver un ``ConjuntoRostros``.
    Cuando este método deje de lanzar ``EtapaPendienteAlumno``,
    ``python main.py lfw`` reutiliza el mismo bucle de experimentos que Olivetti.

    Pasos esperados:

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5, data_home=...)``.
    2. Conservar como máximo 50 identidades, las de mayor frecuencia.
    3. Reindexar ``y`` con ``np.unique(..., return_inverse=True)`` y guardar los nombres.
    4. Leer alto y ancho desde ``images.shape``.
    5. Llamar a ``validar_conjunto`` antes de retornar.
    """

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_lfw_people(
            min_faces_per_person=20,
            resize=0.5,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        
        images = np.asarray(rostros.images)
        x = np.asarray(rostros.data, dtype=float)
        y = np.asarray(rostros.target, dtype=int)
        
        print("Dataset LFW cargado correctamente")
        print(f"Dimensiones de las imagenes: {images.shape}")
        print(f"Dimensiones de x: {x.shape}")
        print(f"Dimensiones de y: {y.shape}")
        print(f"Identidades disponibles: {len(np.unique(y))}")
        
        raise EtapaPendienteAlumno(
            "datos.lfw.CargadorLFW",
            "Carga inicial completada"
            "Pendiente: seleccionar hasta 50 identidades,"
            "reindexar eriquetas y constuir ConjuntoRostros",
        )
