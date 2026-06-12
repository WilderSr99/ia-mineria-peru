
# ============================================================
# inferencia_unet_mineria_local.py
# Inferencia local sobre imagen Sentinel-2 completa
# Modelo: U-Net ResNet34 con 12 canales
#
# Salidas:
#   - probabilidad minería .tif
#   - máscara binaria .tif
#   - polígonos minería .gpkg
# ============================================================

import json
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import segmentation_models_pytorch as smp


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

CONFIG = {
    # Imagen completa a inferir
    "input_raster": r"D:\Geoton_2026\IMAGENES COMPLETAS\para prediccion\S2_AOI_UNIDA_2024_ENERO_DICIEMBRE_DL.tif",

    # Carpeta donde guardaste el modelo descargado desde Drive
    "model_weights": r"D:\Geoton_2026\dataset\resultados\modelo_unet\unet_resnet34_12ch_best_weights.pth",
    "normalization_stats": r"D:\Geoton_2026\dataset\resultados\modelo_unet\normalization_stats.json",

    # Carpeta de salida
    "output_dir": r"D:\Geoton_2026\IMAGENES COMPLETAS\para prediccion\2024_pred",

    # Parámetros
    "tile_size": 256,
    "stride": 128,
    "threshold": 0.50,

    # Si quieres más recall para alertas tempranas, usa 0.40.
    # Si quieres mapa más conservador, usa 0.60.

    # Limpieza vectorial
    "min_area_ha": 0.50,

    # Bandas en el mismo orden del raster
    "band_names": [
        "B2", "B3", "B4", "B8", "B11", "B12",
        "NDVI", "NDWI", "MNDWI", "NBR", "NDBI", "BSI"
    ]
}


# ============================================================
# 2. UTILIDADES
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_norm_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(weights_path, device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=12,
        classes=1,
        activation=None
    )

    state = torch.load(weights_path, map_location=device)

    # Caso 1: archivo es directamente state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    return model


def normalize_image_tile(img, norm_stats, band_names):
    """
    img: numpy array C,H,W
    retorna: numpy array C,H,W en 0-1
    """
    img = img.astype(np.float32)

    for i, band in enumerate(band_names):
        p_low = norm_stats["bands"][band]["p_low"]
        p_high = norm_stats["bands"][band]["p_high"]

        band_arr = img[i]

        band_arr = np.nan_to_num(
            band_arr,
            nan=p_low,
            posinf=p_high,
            neginf=p_low
        )

        band_arr = np.clip(band_arr, p_low, p_high)

        denom = p_high - p_low

        if denom <= 1e-6:
            band_arr = np.zeros_like(band_arr, dtype=np.float32)
        else:
            band_arr = (band_arr - p_low) / (denom + 1e-6)

        band_arr = np.nan_to_num(
            band_arr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        img[i] = band_arr.astype(np.float32)

    return img


def generate_windows_full(width, height, tile_size, stride):
    """
    Genera ventanas cubriendo toda la imagen, incluyendo bordes.
    """
    windows = []

    row_offsets = list(range(0, max(height - tile_size + 1, 1), stride))
    col_offsets = list(range(0, max(width - tile_size + 1, 1), stride))

    if len(row_offsets) == 0:
        row_offsets = [0]
    if len(col_offsets) == 0:
        col_offsets = [0]

    last_row = max(height - tile_size, 0)
    last_col = max(width - tile_size, 0)

    if row_offsets[-1] != last_row:
        row_offsets.append(last_row)
    if col_offsets[-1] != last_col:
        col_offsets.append(last_col)

    for row_off in row_offsets:
        for col_off in col_offsets:
            win_w = min(tile_size, width - col_off)
            win_h = min(tile_size, height - row_off)

            windows.append(Window(col_off, row_off, win_w, win_h))

    return windows


def pad_tile_to_size(tile, tile_size):
    """
    tile: C,H,W
    Si está en borde y mide menos que tile_size, rellena con ceros.
    """
    c, h, w = tile.shape

    if h == tile_size and w == tile_size:
        return tile, h, w

    padded = np.zeros((c, tile_size, tile_size), dtype=np.float32)
    padded[:, :h, :w] = tile

    return padded, h, w


def save_probability_tif(prob_map, reference_raster, output_path):
    with rasterio.open(reference_raster) as src:
        meta = src.meta.copy()

    meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "compress": "lzw",
        "nodata": None
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(prob_map.astype(np.float32), 1)


def save_binary_tif(mask, reference_raster, output_path):
    with rasterio.open(reference_raster) as src:
        meta = src.meta.copy()

    meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 0
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask.astype(np.uint8), 1)


def polygonize_mask(mask_tif, output_gpkg, min_area_ha=0.50):
    """
    Convierte máscara binaria a polígonos GPKG.
    """
    with rasterio.open(mask_tif) as src:
        mask_arr = src.read(1)
        transform = src.transform
        crs = src.crs

        valid_mask = mask_arr == 1

        results = shapes(
            mask_arr.astype(np.uint8),
            mask=valid_mask,
            transform=transform
        )

        geoms = []
        values = []

        for geom, value in results:
            if value == 1:
                geoms.append(shape(geom))
                values.append(int(value))

    if len(geoms) == 0:
        print("[ADVERTENCIA] No se encontraron polígonos.")
        gdf = gpd.GeoDataFrame(
            columns=["clase", "area_m2", "area_ha", "geometry"],
            geometry="geometry",
            crs=crs
        )
        gdf.to_file(output_gpkg, driver="GPKG")
        return gdf

    gdf = gpd.GeoDataFrame(
        {
            "clase": ["mineria"] * len(geoms),
            "value": values
        },
        geometry=geoms,
        crs=crs
    )

    # Calcular área. CRS UTM está en metros, por tanto area está en m2.
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0

    # Filtrar objetos pequeños
    gdf = gdf[gdf["area_ha"] >= min_area_ha].copy()

    # Añadir ID
    gdf["id_pred"] = range(1, len(gdf) + 1)

    # Reordenar columnas
    gdf = gdf[["id_pred", "clase", "value", "area_m2", "area_ha", "geometry"]]

    gdf.to_file(output_gpkg, driver="GPKG")

    return gdf


# ============================================================
# 3. INFERENCIA
# ============================================================

def run_inference(config):
    input_raster = Path(config["input_raster"])
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)

    output_prob_tif = output_dir / "probabilidad_mineria.tif"
    output_mask_tif = output_dir / f"mascara_mineria_thr_{str(config['threshold']).replace('.', '')}.tif"
    output_gpkg = output_dir / f"poligonos_mineria_thr_{str(config['threshold']).replace('.', '')}.gpkg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    norm_stats = load_norm_stats(config["normalization_stats"])
    model = build_model(config["model_weights"], device)

    tile_size = config["tile_size"]
    stride = config["stride"]

    with rasterio.open(input_raster) as src:
        width = src.width
        height = src.height
        count = src.count
        crs = src.crs
        transform = src.transform

        print("[INFO] Raster:", input_raster)
        print("[INFO] CRS:", crs)
        print("[INFO] Tamaño:", width, "x", height)
        print("[INFO] Bandas:", count)
        print("[INFO] Transform:", transform)

        if count != 12:
            raise ValueError(f"El raster debe tener 12 bandas. Tiene {count}.")

        windows = generate_windows_full(width, height, tile_size, stride)

        print("[INFO] Total ventanas:", len(windows))

        prob_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        with torch.no_grad():
            for window in tqdm(windows, desc="Inferencia por tiles"):
                tile = src.read(window=window).astype(np.float32)

                tile_norm = normalize_image_tile(
                    tile,
                    norm_stats=norm_stats,
                    band_names=config["band_names"]
                )

                tile_padded, original_h, original_w = pad_tile_to_size(tile_norm, tile_size)

                x = torch.from_numpy(tile_padded).unsqueeze(0).to(device)

                logits = model(x)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

                # Recortar al tamaño original del borde
                prob = prob[:original_h, :original_w]

                row_off = int(window.row_off)
                col_off = int(window.col_off)

                prob_sum[
                    row_off:row_off + original_h,
                    col_off:col_off + original_w
                ] += prob

                weight_sum[
                    row_off:row_off + original_h,
                    col_off:col_off + original_w
                ] += 1.0

        weight_sum[weight_sum == 0] = 1.0
        prob_map = prob_sum / weight_sum

    binary_mask = (prob_map >= config["threshold"]).astype(np.uint8)

    print("[INFO] Guardando probabilidad:", output_prob_tif)
    save_probability_tif(prob_map, input_raster, output_prob_tif)

    print("[INFO] Guardando máscara binaria:", output_mask_tif)
    save_binary_tif(binary_mask, input_raster, output_mask_tif)

    print("[INFO] Vectorizando máscara:", output_gpkg)
    gdf = polygonize_mask(
        mask_tif=output_mask_tif,
        output_gpkg=output_gpkg,
        min_area_ha=config["min_area_ha"]
    )

    area_total_ha = float(gdf["area_ha"].sum()) if len(gdf) > 0 else 0.0

    summary = {
        "input_raster": str(input_raster),
        "model_weights": config["model_weights"],
        "threshold": config["threshold"],
        "tile_size": tile_size,
        "stride": stride,
        "output_probability_tif": str(output_prob_tif),
        "output_binary_mask_tif": str(output_mask_tif),
        "output_polygons_gpkg": str(output_gpkg),
        "num_polygons": int(len(gdf)),
        "area_total_ha": area_total_ha
    }

    summary_path = output_dir / "inference_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n================ RESUMEN INFERENCIA ================")
    print(json.dumps(summary, indent=4, ensure_ascii=False))
    print("====================================================")

    return summary


if __name__ == "__main__":
    run_inference(CONFIG)