

#- python -m pip uninstall -y transformers huggingface-hub tokenizers safetensors accelerate
#==============
#-# conda install -c conda-forge transformers=4.41.2 huggingface_hub=0.24.6 tokenizers=0.19.1 safetensors=0.4.3 accelerate=0.33.0 -y
#- python -c "from transformers import SegformerForSemanticSegmentation; print('SegFormer OK')"
#(SegFormer OK)
#-python "C:\Users\decg112\Downloads\GEOTON DATOS\DESARROLLO DE LA METOLOGÍA\inferencia_segformer_hf_local.py"




# ============================================================
# inferencia_segformer_hf_local.py
# Inferencia local con SegFormer-B0 12 canales exportado como
# modelo Hugging Face:
#
#   segformer_b0_12ch_hf_model/
#   ├── config.json
#   └── model.safetensors
#
# Entrada:
#   - Imagen Sentinel-2 completa de 12 bandas
#   - Carpeta HF del modelo SegFormer
#   - normalization_stats.json
#
# Salida:
#   - segformer_probabilidad_mineria.tif
#   - segformer_mascara_mineria_thr_050.tif
#   - segformer_poligonos_mineria_thr_050.gpkg
#   - segformer_inference_summary.json
# ============================================================

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


# ============================================================
# 1. CONFIGURACIÓN gaaa
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # Imagen completa de 12 bandas para inferencia
    # Debe tener el mismo orden de bandas usado en entrenamiento:
    # B2, B3, B4, B8, B11, B12, NDVI, NDWI, MNDWI, NBR, NDBI, BSI
    # --------------------------------------------------------

    "input_raster": r"C:\Users\decg112\Downloads\nuevas imagenes_geoton\2025\S2_2025_grid_016_12bandas.tif",

    # --------------------------------------------------------
    # Carpeta del modelo HF exportado desde Colab con:
    # segformer_model.save_pretrained(..., safe_serialization=True)
    # --------------------------------------------------------

    "model_dir": r"D:\Geoton_2026\dataset\resultados\modelo_segformer\segformer_b0_12ch_hf_model",

    # --------------------------------------------------------
    # Normalización usada en entrenamiento
    # --------------------------------------------------------

    "normalization_stats": r"D:\Geoton_2026\dataset\resultados\modelo_segformer\normalization_stats.json",

    # --------------------------------------------------------
    # Carpeta de salida
    # --------------------------------------------------------

    "output_dir": r"C:\Users\decg112\Downloads\nuevas imagenes_geoton\2025\resultado de la prediccion_2025",

    # --------------------------------------------------------
    # Parámetros de inferencia
    # --------------------------------------------------------

    "tile_size": 256,
    "stride": 128,

    # Umbral recomendado:
    # 0.40 = más sensible, más alertas
    # 0.50 = balance general, recomendado para reporte
    # 0.60 = más conservador
    "threshold": 0.50,
    "year" :2025,

    # Filtro mínimo para eliminar polígonos pequeños
    "min_area_ha": 0.50,

    # Orden exacto de bandas
    "band_names": [
        "B2", "B3", "B4", "B8", "B11", "B12",
        "NDVI", "NDWI", "MNDWI", "NBR", "NDBI", "BSI"
    ]
}


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_norm_stats(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe normalization_stats.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_segformer_from_hf_folder(model_dir, device):
    """
    Carga SegFormer desde carpeta local Hugging Face:

        config.json
        model.safetensors

    No descarga nada de Hugging Face.
    No usa .pth.
    """

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del modelo: {model_dir}")

    config_path = model_dir / "config.json"
    safetensors_path = model_dir / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"No existe config.json en: {model_dir}")

    if not safetensors_path.exists():
        raise FileNotFoundError(f"No existe model.safetensors en: {model_dir}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        str(model_dir),
        local_files_only=True
    )

    model = model.to(device)
    model.eval()

    # Validar que el modelo realmente tenga 12 canales.
    # En tu versión de transformers puede estar como encoder o stages.
    in_channels = None

    try:
        first_conv = model.segformer.encoder.patch_embeddings[0].proj
        in_channels = first_conv.in_channels
    except Exception:
        try:
            first_conv = model.segformer.stages[0].patch_embeddings.proj
            in_channels = first_conv.in_channels
        except Exception:
            raise RuntimeError(
                "No se pudo ubicar la primera convolución del SegFormer. "
                "Revisa la versión de transformers."
            )

    print("[OK] Modelo SegFormer cargado desde carpeta local:")
    print("     ", model_dir)
    print("[INFO] Canales de entrada del modelo:", in_channels)

    if in_channels != 12:
        raise RuntimeError(
            f"El modelo cargado no tiene 12 canales de entrada. Tiene: {in_channels}"
        )

    return model


def normalize_image_tile(img, norm_stats, band_names):
    """
    Normaliza un tile con las estadísticas calculadas solo desde train.

    img: numpy array C,H,W
    return: numpy array C,H,W en rango 0-1 aprox.
    """

    img = img.astype(np.float32)

    for i, band in enumerate(band_names):
        if band not in norm_stats["bands"]:
            raise KeyError(f"La banda {band} no existe en normalization_stats.json")

        p_low = norm_stats["bands"][band]["p_low"]
        p_high = norm_stats["bands"][band]["p_high"]

        band_arr = img[i]

        # Reemplazar NaN/Inf antes de normalizar
        band_arr = np.nan_to_num(
            band_arr,
            nan=p_low,
            posinf=p_high,
            neginf=p_low
        )

        # Clip por percentiles de train
        band_arr = np.clip(band_arr, p_low, p_high)

        denom = p_high - p_low

        if denom <= 1e-6:
            band_arr = np.zeros_like(band_arr, dtype=np.float32)
        else:
            band_arr = (band_arr - p_low) / (denom + 1e-6)

        # Seguridad final
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
    Rellena con ceros tiles de borde que tengan menos de tile_size.
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
    Convierte máscara binaria raster a polígonos GPKG.
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
            columns=["id_pred", "clase", "value", "area_m2", "area_ha", "geometry"],
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

    # Si tu CRS está en UTM, el área está en m²
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0

    # Filtrar polígonos pequeños
    gdf = gdf[gdf["area_ha"] >= min_area_ha].copy()

    gdf["id_pred"] = range(1, len(gdf) + 1)

    gdf = gdf[["id_pred", "clase", "value", "area_m2", "area_ha", "geometry"]]

    gdf.to_file(output_gpkg, driver="GPKG")

    return gdf


# ============================================================
# 3. INFERENCIA PRINCIPAL
# ============================================================

def run_inference(config):
    input_raster = Path(config["input_raster"])
    model_dir = Path(config["model_dir"])
    output_dir = Path(config["output_dir"])
    norm_path = Path(config["normalization_stats"])

    ensure_dir(output_dir)

    if not input_raster.exists():
        raise FileNotFoundError(f"No existe el raster de entrada: {input_raster}")

    if not model_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del modelo: {model_dir}")

    if not norm_path.exists():
        raise FileNotFoundError(f"No existe normalization_stats.json: {norm_path}")

    threshold = float(config["threshold"])
    threshold_code = f"{int(threshold * 100):03d}"

    output_prob_tif = output_dir / "segformer_probabilidad_mineria.tif"
    output_mask_tif = output_dir / f"segformer_mascara_mineria_thr_{threshold_code}.tif"
    output_gpkg = output_dir / f"segformer_poligonos_mineria_thr_{threshold_code}.gpkg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("====================================================")
    print("INFERENCIA SEGFORMER-B0 HF LOCAL")
    print("====================================================")
    print("[INFO] Device:", device)

    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))
    else:
        print("[ADVERTENCIA] CUDA no disponible. Se usará CPU.")

    norm_stats = load_norm_stats(norm_path)

    model = build_segformer_from_hf_folder(
        model_dir=model_dir,
        device=device
    )

    tile_size = int(config["tile_size"])
    stride = int(config["stride"])

    with rasterio.open(input_raster) as src:
        width = src.width
        height = src.height
        count = src.count

        print("[INFO] Raster:", input_raster)
        print("[INFO] CRS:", src.crs)
        print("[INFO] Tamaño:", width, "x", height)
        print("[INFO] Bandas:", count)
        print("[INFO] Transform:", src.transform)

        if count != 12:
            raise ValueError(f"El raster debe tener 12 bandas. Tiene {count}.")

        windows = generate_windows_full(width, height, tile_size, stride)

        print("[INFO] Tile size:", tile_size)
        print("[INFO] Stride:", stride)
        print("[INFO] Total ventanas:", len(windows))

        prob_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        with torch.no_grad():
            for window in tqdm(windows, desc="Inferencia SegFormer HF por tiles"):
                tile = src.read(window=window).astype(np.float32)

                tile_norm = normalize_image_tile(
                    tile,
                    norm_stats=norm_stats,
                    band_names=config["band_names"]
                )

                tile_padded, original_h, original_w = pad_tile_to_size(
                    tile_norm,
                    tile_size
                )

                x = torch.from_numpy(tile_padded).unsqueeze(0).to(device)

                outputs = model(pixel_values=x)
                logits = outputs.logits

                logits = F.interpolate(
                    logits,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False
                )

                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

                # Recortar si era tile de borde
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

    prob_min = float(np.nanmin(prob_map))
    prob_max = float(np.nanmax(prob_map))
    prob_mean = float(np.nanmean(prob_map))

    print("[INFO] Probabilidad min:", prob_min)
    print("[INFO] Probabilidad max:", prob_max)
    print("[INFO] Probabilidad mean:", prob_mean)

    binary_mask = (prob_map >= threshold).astype(np.uint8)

    print("[INFO] Guardando probabilidad:", output_prob_tif)
    save_probability_tif(prob_map, input_raster, output_prob_tif)

    print("[INFO] Guardando máscara binaria:", output_mask_tif)
    save_binary_tif(binary_mask, input_raster, output_mask_tif)

    print("[INFO] Vectorizando máscara:", output_gpkg)
    gdf = polygonize_mask(
        mask_tif=output_mask_tif,
        output_gpkg=output_gpkg,
        min_area_ha=float(config["min_area_ha"])
    )

    area_total_ha = float(gdf["area_ha"].sum()) if len(gdf) > 0 else 0.0

    summary = {
        "model": "SegFormer-B0 12ch HF safetensors",
        "input_raster": str(input_raster),
        "model_dir": str(model_dir),
        "threshold": threshold,
        "tile_size": tile_size,
        "stride": stride,
        "prob_min": prob_min,
        "prob_max": prob_max,
        "prob_mean": prob_mean,
        "output_probability_tif": str(output_prob_tif),
        "output_binary_mask_tif": str(output_mask_tif),
        "output_polygons_gpkg": str(output_gpkg),
        "num_polygons": int(len(gdf)),
        "area_total_ha": area_total_ha
    }

    summary_path = output_dir / "segformer_inference_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n================ RESUMEN INFERENCIA SEGFORMER HF ================")
    print(json.dumps(summary, indent=4, ensure_ascii=False))
    print("==================================================================")

    print("\n[OK] Inferencia finalizada.")
    print("[OK] Abrir en QGIS:")
    print("     ", output_prob_tif)
    print("     ", output_mask_tif)
    print("     ", output_gpkg)

    return summary


if __name__ == "__main__":
    run_inference(CONFIG)