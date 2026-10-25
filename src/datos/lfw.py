"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Carga LFW y conserva hasta 50 identidades con mayor frecuencia."""
    
    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_lfw_people(
            min_faces_per_person=20,
            resize=0.5,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        
        images = np.asarray(rostros.images)
        X = np.asarray(rostros.data, dtype=float)
        y = np.asarray(rostros.target, dtype=int)
        
        #Fotografias por identidad y Selección de las 50 más frecuentes
        identidades, cantidades = np.unique(y, return_counts=True)
        
        orden = np.argsort(-cantidades, kind="stable")
        seleccionadas = identidades[orden[:50]]
        
        #Filtrar fotografias y Etiquetas
        mascara = np.isin(y, seleccionadas)
        
        images = images[mascara]
        X = X[mascara]
        y = y[mascara]
        
        #Reindexar y Conservar Nombres
        identidades_originales, y = np.unique(y, return_inverse=True)
        nombres = [str(rostros.target_names[i]) for i in identidades_originales]

        #Dimensiones de las imagenes
        alto = int(images.shape[1])
        ancho = int(images.shape[2])
        
        #Construir, validar y devolver conjunto rostro
        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho
        )
        
        validar_conjunto(conjunto)
        return conjunto