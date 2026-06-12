# ============================================================
# preparar_dataset_mineria.py
# Preparación local de dataset para segmentación de minería
# Sentinel-2 multibanda + polígonos manuales GPKG
#
# Versión:
#   - Soporta varios años: 2019, 2020, 2025
#   - Genera máscaras rasterizadas
#   - Genera tiles imagen/máscara
#   - Aplica split estratificado por:
#       año + is_positive
#   - Calcula estadísticas de normalización usando SOLO train
#
# Entrada:
#   - Raster Sentinel-2 con 12 bandas por año
#   - GPKG manual reproyectado al CRS del raster por año
#
# Salida:
#   - full_masks/mask_YYYY.tif
#   - images/train, images/val, images/test
#   - masks/train, masks/val, masks/test
#   - metadata/tiles_metadata.csv
#   - metadata/config_dataset.json
#   - metadata/clases.json
#   - metadata/normalization_stats.json
# ============================================================

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # RUTAS DE ENTRADA
    # Agrega más años aquí si luego tienes 2021, 2022, etc.
    # --------------------------------------------------------

    "datasets": [
        {
            "year": 2019,
            "raster": r"D:\Geoton_2026\IMAGENES COMPLETAS\S2_AOI_UNIDA_2019_M6_10_DL.tif",
            "gpkg": r"D:\Geoton_2026\IMAGENES COMPLETAS\CRS CORREGIDAS\mineria_manual_2019_reproyectado.gpkg",
            "layer": None
        },
        {
            "year": 2020,
            "raster": r"D:\Geoton_2026\IMAGENES COMPLETAS\S2_AOI_UNIDA_2020_M6_10_DL.tif",
            "gpkg": r"D:\Geoton_2026\IMAGENES COMPLETAS\CRS CORREGIDAS\mineria_manual_2020_reproyectado.gpkg",
            "layer": None
        },
        {
            "year": 2025,
            "raster": r"D:\Geoton_2026\IMAGENES COMPLETAS\S2_AOI_UNIDA_2025_M6_10_DL.tif",
            "gpkg": r"D:\Geoton_2026\IMAGENES COMPLETAS\CRS CORREGIDAS\mineria_manual_2025_reproyectado.gpkg",
            "layer": None
        }
    ],

    # --------------------------------------------------------
    # CARPETA DE SALIDA
    # --------------------------------------------------------

    "output_dir": r"D:\Geoton_2026\dataset_mineria_sentinel",

    # --------------------------------------------------------
    # BANDAS DEL RASTER
    # Deben estar en el mismo orden en que fueron exportadas desde GEE.
    # --------------------------------------------------------

    "band_names": [
        "B2",      # Blue, 10 m
        "B3",      # Green, 10 m
        "B4",      # Red, 10 m
        "B8",      # NIR, 10 m
        "B11",     # SWIR1, 20 m reescalado a 10 m
        "B12",     # SWIR2, 20 m reescalado a 10 m
        "NDVI",
        "NDWI",
        "MNDWI",
        "NBR",
        "NDBI",
        "BSI"
    ],

    # --------------------------------------------------------
    # TIPO DE MÁSCARA
    #
    # binary:
    #   0 = fondo / no minería
    #   1 = minería
    #
    # multiclass:
    #   0 = fondo
    #   1 = poza_mineria
    #   2 = suelo_expuesto
    #   3 = area_degradada
    # --------------------------------------------------------

    "mask_mode": "binary",

    "field_clase": "clase",
    "field_tipo_area": "tipo_area",

    "multiclass_map": {
        "poza_mineria": 1,
        "suelo_expuesto": 2,
        "area_degradada": 3
    },

    # --------------------------------------------------------
    # TILEADO
    # --------------------------------------------------------

    # Sentinel-2 a 10 m:
    # 256 px = 2.56 km x 2.56 km aproximadamente.
    "tile_size": 256,

    # 128 = 50 % de solape.
    "stride": 128,

    # Por cada tile positivo se intentará guardar hasta esta proporción de negativos.
    # Si no hay suficientes negativos, se guardan todos los negativos disponibles.
    "negative_ratio": 2.0,

    # Fracción mínima de píxeles mineros para considerar tile positivo.
    # 0.0001 = 0.01 %
    "min_positive_fraction": 0.0001,

    # Fracción mínima de píxeles válidos en el tile.
    "min_valid_fraction": 0.70,

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    "split": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15
    },

    # Modo de split:
    #   "stratified" = recomendado ahora: año + positivo/negativo
    #   "spatial"    = split por bloque espacial
    "split_mode": "stratified",

    # Se conserva para registrar bloque espacial y para usar split_mode="spatial".
    "spatial_block_size": 1024,

    # --------------------------------------------------------
    # NORMALIZACIÓN
    # No se aplica físicamente a los GeoTIFF.
    # Se calculan estadísticas solo con tiles de train.
    # --------------------------------------------------------

    "normalization": {
        "method": "percentile_clip",
        "lower_percentile": 2,
        "upper_percentile": 98,
        "sample_pixels_per_tile": 5000,
        "max_tiles_for_stats": 1000
    },

    # --------------------------------------------------------
    # CONTROL
    # --------------------------------------------------------

    "seed": 42,

    # Si True, elimina la carpeta de salida si ya existe.
    "overwrite_output": True,

    # Si True, marca como minería todo píxel tocado por el polígono.
    # Si en QGIS ves que la máscara se expande demasiado, cambia a False.
    "all_touched": True
}


# ============================================================
# 2. FUNCIONES DE CARPETAS Y VALIDACIÓN
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def prepare_output_dir(output_dir, overwrite=True):
    output_dir = Path(output_dir)

    if output_dir.exists() and overwrite:
        print(f"[INFO] Eliminando carpeta previa: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "masks" / split)

    ensure_dir(output_dir / "full_masks")
    ensure_dir(output_dir / "metadata")


def read_gpkg(gpkg_path, layer=None):
    if layer is None:
        return gpd.read_file(gpkg_path)
    return gpd.read_file(gpkg_path, layer=layer)


def validate_raster(raster_path, band_names):
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"No existe el raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        print("\n===================================================")
        print(f"[RASTER] {raster_path}")
        print("===================================================")
        print(f"CRS:        {src.crs}")
        print(f"Tamaño:     {src.width} x {src.height}")
        print(f"Bandas:     {src.count}")
        print(f"Resolución: {src.res}")
        print(f"Dtype:      {src.dtypes}")
        print(f"Nodata:     {src.nodata}")
        print(f"Bounds:     {src.bounds}")

        if src.count != len(band_names):
            raise ValueError(
                f"El raster tiene {src.count} bandas, "
                f"pero configuraste {len(band_names)} nombres de banda."
            )

        if src.crs is None:
            raise ValueError("El raster no tiene CRS definido.")

        return {
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "bounds": src.bounds,
            "res": src.res,
            "nodata": src.nodata
        }


def validate_gpkg_against_raster(gpkg_path, raster_path, layer=None):
    gpkg_path = Path(gpkg_path)

    if not gpkg_path.exists():
        raise FileNotFoundError(f"No existe el GPKG: {gpkg_path}")

    gdf = read_gpkg(gpkg_path, layer=layer)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        b = src.bounds
        raster_bbox = box(b.left, b.bottom, b.right, b.top)

    print("\n===================================================")
    print(f"[GPKG] {gpkg_path}")
    print("===================================================")
    print(f"CRS:       {gdf.crs}")
    print(f"Registros: {len(gdf)}")
    print(f"Campos:    {list(gdf.columns)}")

    if gdf.crs is None:
        raise ValueError("El GPKG no tiene CRS definido.")

    if gdf.crs != raster_crs:
        raise ValueError(
            "El CRS del GPKG no coincide con el CRS del raster. "
            "Ejecuta primero el script 00_validar_corregir_crs.py."
        )

    gdf = gdf[gdf.geometry.notnull()].copy()

    invalidas = int((~gdf.is_valid).sum())
    if invalidas > 0:
        print(f"[ADVERTENCIA] Geometrías inválidas: {invalidas}")
        print("[INFO] Corrigiendo con buffer(0)")
        gdf["geometry"] = gdf.geometry.buffer(0)

    intersecta = gdf.intersects(raster_bbox)
    n_intersecta = int(intersecta.sum())

    print(f"Polígonos que intersectan el raster: {n_intersecta} / {len(gdf)}")

    if n_intersecta == 0:
        raise ValueError(
            "Ningún polígono intersecta el raster. "
            "Revisa CRS, extensión del raster o área de estudio."
        )

    gdf = gdf[intersecta].copy()

    return gdf


# ============================================================
# 3. RASTERIZACIÓN DE GPKG A MÁSCARA
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def build_rasterization_shapes(gdf, config):
    mask_mode = config["mask_mode"]

    if mask_mode == "binary":
        shapes = []

        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                shapes.append((geom, 1))

        return shapes

    elif mask_mode == "multiclass":
        field_tipo = config["field_tipo_area"]
        class_map = config["multiclass_map"]

        if field_tipo not in gdf.columns:
            raise ValueError(f"No existe el campo '{field_tipo}' en el GPKG.")

        shapes = []

        for _, row in gdf.iterrows():
            geom = row.geometry
            tipo = normalize_text(row[field_tipo])

            if geom is None or geom.is_empty:
                continue

            value = class_map.get(tipo, 0)

            if value > 0:
                shapes.append((geom, value))

        return shapes

    else:
        raise ValueError("mask_mode debe ser 'binary' o 'multiclass'.")


def rasterize_gpkg_to_mask(raster_path, gpkg_path, output_mask_path, config, layer=None):
    print("\n===================================================")
    print(f"[RASTERIZANDO] {gpkg_path}")
    print("===================================================")

    gdf = validate_gpkg_against_raster(gpkg_path, raster_path, layer=layer)
    shapes = build_rasterization_shapes(gdf, config)

    if len(shapes) == 0:
        raise ValueError("No hay geometrías válidas para rasterizar.")

    with rasterio.open(raster_path) as src:
        mask = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
            all_touched=config["all_touched"]
        )

        mask_meta = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "compress": "lzw",
            "nodata": 0
        }

    output_mask_path = Path(output_mask_path)
    ensure_dir(output_mask_path.parent)

    with rasterio.open(output_mask_path, "w", **mask_meta) as dst:
        dst.write(mask, 1)

    unique, counts = np.unique(mask, return_counts=True)

    print("[INFO] Distribución de máscara:")
    for u, c in zip(unique, counts):
        print(f"Clase {u}: {c:,} píxeles")

    print(f"[OK] Máscara guardada: {output_mask_path}")

    return output_mask_path


# ============================================================
# 4. TILEADO Y SPLIT
# ============================================================

def generate_windows(width, height, tile_size, stride):
    windows = []

    for row_off in range(0, height - tile_size + 1, stride):
        for col_off in range(0, width - tile_size + 1, stride):
            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=tile_size,
                    height=tile_size
                )
            )

    return windows


def get_valid_fraction(image_array, nodata=None):
    """
    image_array: shape = (bands, height, width)
    """

    if nodata is not None:
        valid = np.all(image_array != nodata, axis=0)
    else:
        valid = np.all(np.isfinite(image_array), axis=0)

    return float(valid.mean())


def get_positive_fraction(mask_array):
    return float((mask_array > 0).mean())


def get_spatial_block_id(window, spatial_block_size):
    block_x = int(window.col_off // spatial_block_size)
    block_y = int(window.row_off // spatial_block_size)
    return f"{block_x}_{block_y}"


def assign_split_by_block(block_id, split_config, seed):
    """
    Split espacial opcional.
    Para esta versión se recomienda split_mode='stratified'.
    """

    rnd = random.Random(f"{seed}_{block_id}").random()

    train_p = split_config["train"]
    val_p = split_config["val"]

    if rnd < train_p:
        return "train"
    elif rnd < train_p + val_p:
        return "val"
    else:
        return "test"


def collect_tile_candidates(raster_path, mask_path, year, config):
    print("\n===================================================")
    print(f"[BUSCANDO TILES] Año {year}")
    print("===================================================")

    tile_size = config["tile_size"]
    stride = config["stride"]

    positive_items = []
    negative_items = []

    with rasterio.open(raster_path) as src, rasterio.open(mask_path) as msrc:
        if src.width != msrc.width or src.height != msrc.height:
            raise ValueError("Raster y máscara no tienen el mismo ancho/alto.")

        if src.transform != msrc.transform:
            raise ValueError("Raster y máscara no tienen el mismo transform.")

        windows = generate_windows(
            width=src.width,
            height=src.height,
            tile_size=tile_size,
            stride=stride
        )

        print(f"[INFO] Ventanas generadas: {len(windows)}")

        for idx, window in enumerate(tqdm(windows, desc=f"Explorando {year}")):
            img = src.read(window=window)
            mask = msrc.read(1, window=window)

            valid_fraction = get_valid_fraction(img, nodata=src.nodata)

            if valid_fraction < config["min_valid_fraction"]:
                continue

            positive_fraction = get_positive_fraction(mask)

            block_id = get_spatial_block_id(
                window=window,
                spatial_block_size=config["spatial_block_size"]
            )

            is_positive = positive_fraction >= config["min_positive_fraction"]

            item = {
                "year": int(year),
                "window_index": int(idx),
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
                "positive_fraction": float(positive_fraction),
                "valid_fraction": float(valid_fraction),
                "block_id": block_id,
                "is_positive": bool(is_positive),

                # Se asigna después.
                "split": None
            }

            if is_positive:
                positive_items.append(item)
            else:
                negative_items.append(item)

    print(f"[INFO] Tiles positivos: {len(positive_items)}")
    print(f"[INFO] Tiles negativos candidatos: {len(negative_items)}")

    return positive_items, negative_items


def sample_negative_tiles(positive_items, negative_items, negative_ratio, seed):
    """
    Selecciona negativos para no llenar el dataset de fondo.
    Si no hay suficientes negativos, guarda todos.
    """

    n_pos = len(positive_items)
    n_neg_target = int(n_pos * negative_ratio)

    random.seed(seed)

    if n_pos == 0:
        print("[ADVERTENCIA] No hay tiles positivos. Se guardará una muestra pequeña de negativos.")
        n_neg_target = min(len(negative_items), 500)

    if len(negative_items) <= n_neg_target:
        sampled_negatives = negative_items
    else:
        sampled_negatives = random.sample(negative_items, n_neg_target)

    selected = positive_items + sampled_negatives
    random.shuffle(selected)

    print(f"[INFO] Positivos seleccionados: {len(positive_items)}")
    print(f"[INFO] Negativos seleccionados: {len(sampled_negatives)}")
    print(f"[INFO] Total seleccionado: {len(selected)}")

    return selected


def assign_stratified_split(items, split_config, seed):
    """
    Asigna train/val/test manteniendo proporciones por:
    - year
    - is_positive

    Esto evita que todos los negativos caigan en test o que train quede
    casi sin negativos.
    """

    print("\n===================================================")
    print("[SPLIT ESTRATIFICADO] year + is_positive")
    print("===================================================")

    df = pd.DataFrame(items)

    if "year" not in df.columns or "is_positive" not in df.columns:
        raise ValueError("Los items deben tener columnas 'year' e 'is_positive'.")

    train_p = split_config["train"]
    val_p = split_config["val"]

    assigned_items = []
    rng = random.Random(seed)

    for (year, is_positive), group in df.groupby(["year", "is_positive"]):
        group_records = group.to_dict("records")
        rng.shuffle(group_records)

        n = len(group_records)

        n_train = int(round(n * train_p))
        n_val = int(round(n * val_p))

        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        train_records = group_records[:n_train]
        val_records = group_records[n_train:n_train + n_val]
        test_records = group_records[n_train + n_val:]

        for item in train_records:
            item["split"] = "train"
            assigned_items.append(item)

        for item in val_records:
            item["split"] = "val"
            assigned_items.append(item)

        for item in test_records:
            item["split"] = "test"
            assigned_items.append(item)

        print(
            f"Año {year} | is_positive={is_positive} | "
            f"total={n} → train={len(train_records)}, "
            f"val={len(val_records)}, test={len(test_records)}"
        )

    rng.shuffle(assigned_items)

    return assigned_items


# ============================================================
# 5. ESCRITURA DE TILES
# ============================================================

def write_image_tile(src, window, output_path):
    tile = src.read(window=window)

    meta = src.meta.copy()
    meta.update({
        "height": int(window.height),
        "width": int(window.width),
        "transform": src.window_transform(window),
        "compress": "lzw"
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(tile)


def write_mask_tile(msrc, window, output_path):
    mask = msrc.read(1, window=window).astype(np.uint8)

    meta = msrc.meta.copy()
    meta.update({
        "height": int(window.height),
        "width": int(window.width),
        "transform": msrc.window_transform(window),
        "compress": "lzw",
        "dtype": "uint8",
        "count": 1,
        "nodata": 0
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask, 1)


def create_tiles_from_candidates(selected_items, output_dir):
    print("\n===================================================")
    print("[ESCRIBIENDO TILES FINALES]")
    print("===================================================")

    output_dir = Path(output_dir)
    records = []

    df_items = pd.DataFrame(selected_items)

    required_cols = ["year", "raster_path", "mask_full_path", "split"]
    for col in required_cols:
        if col not in df_items.columns:
            raise ValueError(f"Falta la columna requerida en selected_items: {col}")

    grouped = df_items.groupby(["year", "raster_path", "mask_full_path"])

    for (year, raster_path, mask_full_path), group in grouped:
        year = int(year)

        print(f"\n[INFO] Escribiendo tiles del año {year}")
        print(f"[INFO] Raster: {raster_path}")
        print(f"[INFO] Máscara completa: {mask_full_path}")

        group_records = group.to_dict("records")

        with rasterio.open(raster_path) as src, rasterio.open(mask_full_path) as msrc:
            for _, item in enumerate(tqdm(group_records, desc=f"Guardando tiles {year}")):
                split = item["split"]

                window = Window(
                    col_off=int(item["col_off"]),
                    row_off=int(item["row_off"]),
                    width=int(item["width"]),
                    height=int(item["height"])
                )

                tile_id = f"{year}_{int(item['window_index']):06d}"

                image_name = f"tile_{tile_id}.tif"
                mask_name = f"tile_{tile_id}_mask.tif"

                image_output = output_dir / "images" / split / image_name
                mask_output = output_dir / "masks" / split / mask_name

                ensure_dir(image_output.parent)
                ensure_dir(mask_output.parent)

                write_image_tile(src, window, image_output)
                write_mask_tile(msrc, window, mask_output)

                record = item.copy()
                record.update({
                    "tile_id": tile_id,
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "image_path": str(image_output),
                    "mask_path": str(mask_output),
                    "is_positive": bool(item["is_positive"])
                })

                records.append(record)

    return records


# ============================================================
# 6. ESTADÍSTICAS DE NORMALIZACIÓN
# ============================================================

def compute_normalization_stats(output_dir, band_names, config):
    print("\n===================================================")
    print("[NORMALIZACIÓN] Calculando estadísticas usando SOLO train")
    print("===================================================")

    train_dir = Path(output_dir) / "images" / "train"
    train_images = sorted(train_dir.glob("*.tif"))

    if len(train_images) == 0:
        raise ValueError("No existen imágenes en images/train.")

    norm_cfg = config["normalization"]

    max_tiles = min(len(train_images), norm_cfg["max_tiles_for_stats"])
    sampled_paths = random.sample(train_images, max_tiles)

    sample_pixels_per_tile = norm_cfg["sample_pixels_per_tile"]

    band_values = [[] for _ in band_names]

    for path in tqdm(sampled_paths, desc="Muestreando train"):
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)

        bands, h, w = arr.shape
        flat = arr.reshape(bands, -1)

        n_pixels = flat.shape[1]
        n_sample = min(sample_pixels_per_tile, n_pixels)

        idx = np.random.choice(n_pixels, size=n_sample, replace=False)

        for b in range(bands):
            vals = flat[b, idx]
            vals = vals[np.isfinite(vals)]
            band_values[b].append(vals)

    stats = {
        "important_note": "La normalización debe aplicarse durante el entrenamiento, no se aplicó a los GeoTIFF.",
        "computed_from": "train_only",
        "method": norm_cfg["method"],
        "lower_percentile": norm_cfg["lower_percentile"],
        "upper_percentile": norm_cfg["upper_percentile"],
        "bands": {}
    }

    for b, band_name in enumerate(band_names):
        vals = np.concatenate(band_values[b])

        p_low = float(np.percentile(vals, norm_cfg["lower_percentile"]))
        p_high = float(np.percentile(vals, norm_cfg["upper_percentile"]))
        mean = float(np.mean(vals))
        std = float(np.std(vals))

        stats["bands"][band_name] = {
            "p_low": p_low,
            "p_high": p_high,
            "mean": mean,
            "std": std
        }

    stats_path = Path(output_dir) / "metadata" / "normalization_stats.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"[OK] Estadísticas guardadas en: {stats_path}")

    return stats


# ============================================================
# 7. METADATA Y README
# ============================================================

def save_metadata(records, output_dir, config):
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "metadata"
    ensure_dir(metadata_dir)

    df = pd.DataFrame(records)

    metadata_csv = metadata_dir / "tiles_metadata.csv"
    df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    config_path = metadata_dir / "config_dataset.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    clases = {
        "mask_mode": config["mask_mode"],
        "binary": {
            "0": "fondo_no_mineria",
            "1": "mineria"
        },
        "multiclass": {
            "0": "fondo_no_mineria",
            "1": "poza_mineria",
            "2": "suelo_expuesto",
            "3": "area_degradada"
        }
    }

    clases_path = metadata_dir / "clases.json"
    with open(clases_path, "w", encoding="utf-8") as f:
        json.dump(clases, f, indent=4, ensure_ascii=False)

    years = [str(ds["year"]) for ds in config["datasets"]]

    readme_path = output_dir / "README_dataset.txt"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("DATASET MINERÍA SENTINEL-2\n")
        f.write("==========================\n\n")
        f.write("Dataset preparado para segmentación semántica de minería.\n\n")
        f.write(f"Años incluidos: {', '.join(years)}.\n")
        f.write("Entrada: Sentinel-2 multibanda + índices espectrales.\n")
        f.write("Máscara: rasterizada desde polígonos manuales GPKG.\n\n")

        f.write("Bandas de entrada:\n")
        for i, band in enumerate(config["band_names"], start=1):
            f.write(f"{i}. {band}\n")

        f.write("\nModo de máscara:\n")
        f.write(f"{config['mask_mode']}\n\n")

        f.write("Clases binarias:\n")
        f.write("0 = fondo / no minería\n")
        f.write("1 = minería\n\n")

        f.write("Split:\n")
        f.write(f"Modo de split: {config.get('split_mode', 'spatial')}\n")
        f.write("Si es stratified, se estratifica por año + is_positive.\n\n")

        f.write("IMPORTANTE SOBRE NORMALIZACIÓN:\n")
        f.write("Los GeoTIFF no fueron normalizados físicamente.\n")
        f.write("Se calcularon estadísticas usando SOLO tiles de train.\n")
        f.write("Aplicar la normalización dentro del Dataset de PyTorch en Colab.\n")

    print(f"[OK] Metadata guardada en: {metadata_dir}")


def print_summary(records):
    df = pd.DataFrame(records)

    print("\n===================================================")
    print("RESUMEN FINAL DEL DATASET")
    print("===================================================")

    print(f"Total tiles: {len(df)}")

    print("\nTiles por split:")
    print(df["split"].value_counts())

    print("\nTiles por año:")
    print(df["year"].value_counts())

    print("\nAño x split:")
    print(pd.crosstab(df["year"], df["split"]))

    print("\nPositivos por split:")
    print(pd.crosstab(df["split"], df["is_positive"]))

    print("\nPositivos por año y split:")
    print(pd.crosstab([df["year"], df["split"]], df["is_positive"]))

    print("\nPromedio positive_fraction por split:")
    print(df.groupby("split")["positive_fraction"].mean())

    print("\nPromedio positive_fraction por año:")
    print(df.groupby("year")["positive_fraction"].mean())

    print("===================================================")


# ============================================================
# 8. PIPELINE PRINCIPAL
# ============================================================

def main():
    config = CONFIG

    random.seed(config["seed"])
    np.random.seed(config["seed"])

    output_dir = Path(config["output_dir"])

    prepare_output_dir(
        output_dir=output_dir,
        overwrite=config["overwrite_output"]
    )

    # --------------------------------------------------------
    # Validar todos los rasters
    # --------------------------------------------------------

    for ds in config["datasets"]:
        validate_raster(ds["raster"], config["band_names"])

    # --------------------------------------------------------
    # Rasterizar máscaras y recolectar candidatos
    # --------------------------------------------------------

    all_selected_items = []

    for ds in config["datasets"]:
        year = int(ds["year"])
        raster_path = ds["raster"]
        gpkg_path = ds["gpkg"]
        layer = ds["layer"]

        print("\n###################################################")
        print(f"PROCESANDO AÑO {year}")
        print("###################################################")

        mask_full_path = output_dir / "full_masks" / f"mask_{year}.tif"

        rasterize_gpkg_to_mask(
            raster_path=raster_path,
            gpkg_path=gpkg_path,
            output_mask_path=mask_full_path,
            config=config,
            layer=layer
        )

        positive_items, negative_items = collect_tile_candidates(
            raster_path=raster_path,
            mask_path=mask_full_path,
            year=year,
            config=config
        )

        selected_items = sample_negative_tiles(
            positive_items=positive_items,
            negative_items=negative_items,
            negative_ratio=config["negative_ratio"],
            seed=config["seed"] + year
        )

        # Guardamos rutas dentro de cada item para escribir tiles después del split.
        for item in selected_items:
            item["raster_path"] = str(raster_path)
            item["mask_full_path"] = str(mask_full_path)

        all_selected_items.extend(selected_items)

    if len(all_selected_items) == 0:
        raise ValueError("No se generó ningún tile candidato. Revisa máscaras, CRS, tile_size o stride.")

    # --------------------------------------------------------
    # Asignar split final
    # --------------------------------------------------------

    split_mode = config.get("split_mode", "spatial")

    if split_mode == "stratified":
        all_selected_items = assign_stratified_split(
            items=all_selected_items,
            split_config=config["split"],
            seed=config["seed"]
        )
    elif split_mode == "spatial":
        print("\n===================================================")
        print("[SPLIT ESPACIAL] block_id")
        print("===================================================")

        for item in all_selected_items:
            item["split"] = assign_split_by_block(
                block_id=item["block_id"],
                split_config=config["split"],
                seed=config["seed"]
            )
    else:
        raise ValueError("split_mode debe ser 'stratified' o 'spatial'.")

    # --------------------------------------------------------
    # Escribir tiles físicos
    # --------------------------------------------------------

    records = create_tiles_from_candidates(
        selected_items=all_selected_items,
        output_dir=output_dir
    )

    if len(records) == 0:
        raise ValueError("No se escribió ningún tile final.")

    # --------------------------------------------------------
    # Guardar metadata
    # --------------------------------------------------------

    save_metadata(
        records=records,
        output_dir=output_dir,
        config=config
    )

    # --------------------------------------------------------
    # Calcular estadísticas de normalización solo con train
    # --------------------------------------------------------

    compute_normalization_stats(
        output_dir=output_dir,
        band_names=config["band_names"],
        config=config
    )

    # --------------------------------------------------------
    # Resumen final
    # --------------------------------------------------------

    print_summary(records)

    print("\n[OK] Dataset preparado correctamente.")
    print("[OK] Split aplicado:", split_mode)
    print(f"[OK] Carpeta final: {output_dir}")


if __name__ == "__main__":
    main()

## Para comprimir 
"""
Compress-Archive `
  -Path "D:\Geoton_2026\dataset_mineria_sentinel" `
  -DestinationPath "D:\Geoton_2026\dataset_mineria_sentinel_v1_stratified.zip" `
  -Force

"""