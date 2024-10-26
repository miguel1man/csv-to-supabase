# import_songs.py

import pandas as pd
import numpy as np
from supabase import create_client
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Any
import json
import csv
from pathlib import Path
import chardet

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SongDataImporter:
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Inicializa el importador con las credenciales de Supabase.

        Args:
            supabase_url: URL de la instancia de Supabase
            supabase_key: Clave de acceso a Supabase
        """
        self.supabase = create_client(supabase_url, supabase_key)
        self.failed_records: List[Dict] = (
            []
        )  # Inicialización de la lista de registros fallidos

    def clean_number(self, value: Any) -> Optional[int]:
        """
        Limpia valores numéricos, removiendo comas y otros caracteres no numéricos.
        """
        if pd.isna(value) or value == "" or value is None:
            return None

        try:
            # Si es un string, eliminar comas y convertir a número
            if isinstance(value, str):
                # Eliminar comas y espacios
                cleaned = value.replace(",", "").replace(" ", "")
                return int(float(cleaned))
            # Si ya es un número, convertirlo directamente
            elif isinstance(value, (int, float, np.number)):
                return int(value)
            return None
        except (ValueError, TypeError):
            logger.warning(f"No se pudo convertir el valor '{value}' a número")
            return None

    def clean_value(self, value: Any, field_name: str = None) -> Optional[Any]:
        """
        Limpia y valida valores para asegurar que son compatibles con JSON.
        """
        if pd.isna(value) or value == "" or value is None:
            return None

        # Manejo especial para youtube_views
        if field_name == "youtube_views":
            return self.clean_number(value)

        if isinstance(value, (float, np.float64)):
            return float(value) if not np.isnan(value) else None
        if isinstance(value, (int, np.int64)):
            return int(value)
        return str(value).strip()

    def prepare_song_data(self, row: pd.Series) -> Dict:
        """
        Prepara los datos de una canción para su inserción.
        """
        song_data = {
            "title": self.clean_value(row["title"], "title"),
            "artist": self.clean_value(row["artist"], "artist"),
            "average_score": self.clean_value(row["average_score"], "average_score"),
            "score_2024_10": self.clean_value(row["score_2024_10"], "score_2024_10"),
            "score_2024_q3": self.clean_value(row["score_2024_q3"], "score_2024_q3"),
            "score_2024_q2": self.clean_value(row["score_2024_q2"], "score_2024_q2"),
            "score_2024_q1": self.clean_value(row["score_2024_q1"], "score_2024_q1"),
            "score_2023": self.clean_value(row["score_2023"], "score_2023"),
            "album_date": (
                str(row["album_date"]) if not pd.isna(row["album_date"]) else None
            ),
            "language": self.clean_value(row["language"], "language"),
            "genre": self.clean_value(row["genre"], "genre"),
            "playlists_name": self.clean_value(row["playlists_name"], "playlists_name"),
            "energy": self.clean_value(row["energy"], "energy"),
            "youtube_url": self.clean_value(row["youtube_url"], "youtube_url"),
            "youtube_views": self.clean_value(row["youtube_views"], "youtube_views"),
            "spotify_url": self.clean_value(row["spotify_url"], "spotify_url"),
            "album_name": self.clean_value(row["album_name"], "album_name"),
        }

        # Elimina las claves con valores None
        return {k: v for k, v in song_data.items() if v is not None}

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Lee el archivo CSV con los datos de las canciones.
        """
        try:
            # Detectar codificación
            with open(file_path, "rb") as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result["encoding"]
                logger.info(
                    f"Codificación detectada: {encoding} con confianza: {result['confidence']}"
                )

            # Leer CSV con la codificación detectada
            df = pd.read_csv(file_path, sep=";", encoding=encoding)

            # Limpia los nombres de las columnas
            df.columns = df.columns.str.lower().str.strip()

            # Convierte la fecha del álbum a formato datetime
            df["album_date"] = pd.to_datetime(df["album_date"]).dt.date

            # Reemplaza valores NaN con None
            df = df.replace({np.nan: None})

            # Verifica los datos
            logger.info(f"Muestra de datos leídos:")
            logger.info(df.head(1).to_string())
            logger.info(f"CSV leído exitosamente. {len(df)} registros encontrados.")

            return df

        except Exception as e:
            logger.error(f"Error al leer el CSV: {str(e)}")
            raise

    def save_failed_records(self):
        """
        Guarda los registros fallidos en archivos CSV y JSON.
        """
        if not self.failed_records:  # Si no hay registros fallidos, no hacer nada
            logger.info("No hay registros fallidos para guardar.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Crear directorio para los logs si no existe
        log_dir = Path("failed_imports")
        log_dir.mkdir(exist_ok=True)

        # Guardar en formato CSV
        csv_path = log_dir / f"failed_records_{timestamp}.csv"
        df_failed = pd.DataFrame(self.failed_records)
        df_failed.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")

        # Guardar en formato JSON para referencia
        json_path = log_dir / f"failed_records_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.failed_records, f, ensure_ascii=False, indent=2)

        logger.info(
            f"""
        Registros fallidos guardados en:
        - CSV: {csv_path}
        - JSON: {json_path}
        Total de registros fallidos: {len(self.failed_records)}
        """
        )

    def import_songs(self, df: pd.DataFrame) -> None:
        """
        Importa las canciones a Supabase.
        """
        successful_imports = 0
        failed_imports = 0
        total_rows = len(df)

        try:
            for index, row in df.iterrows():
                try:
                    # Prepara los datos
                    song_data = self.prepare_song_data(row)

                    logger.info(
                        f"Intentando insertar canción ({index + 1}/{total_rows}): {song_data['title']}"
                    )

                    # Intenta insertar la canción
                    result = self.supabase.table("songs").insert(song_data).execute()

                    logger.info(
                        f"Canción '{song_data['title']}' importada exitosamente."
                    )
                    successful_imports += 1

                except Exception as e:
                    logger.error(f"Error al importar canción {index + 1}: {str(e)}")
                    logger.error(f"Datos problemáticos: {song_data}")

                    # Guardar el registro fallido con información adicional
                    failed_record = row.to_dict()
                    failed_record["error_message"] = str(e)
                    failed_record["error_time"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    self.failed_records.append(failed_record)

                    failed_imports += 1
                    continue

        except Exception as e:
            logger.error(f"Error al importar canciones: {str(e)}")
            raise
        finally:
            # Generar el archivo de registros fallidos
            self.save_failed_records()

            logger.info(
                f"""
            Resumen de importación:
            - Total de registros procesados: {total_rows}
            - Importaciones exitosas: {successful_imports}
            - Importaciones fallidas: {failed_imports}
            """
            )


def main():
    try:
        # Cargar variables de entorno
        load_dotenv()

        # Obtener credenciales
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not all([supabase_url, supabase_key]):
            raise ValueError(
                "Las credenciales de Supabase no están configuradas correctamente."
            )

        # Crear instancia del importador
        importer = SongDataImporter(supabase_url, supabase_key)

        # Leer y procesar el CSV
        csv_path = "songs2.csv"  # Actualiza esta ruta
        df = importer.read_csv(csv_path)

        # Importar datos
        importer.import_songs(df)

        logger.info("Proceso de importación completado.")

    except Exception as e:
        logger.error(f"Error durante la importación: {str(e)}")
        raise


if __name__ == "__main__":
    main()
