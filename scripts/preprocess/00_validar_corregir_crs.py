# ============================================================
# 00_VALIDAR_CORREGIR_CRS.py
# Validación y corrección de CRS para Sentinel-2 + GPKG
# ============================================================

from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box


# ============================================================
# 1. CONFIGURA TUS RUTAS
# ============================================================

raster_2019 = r"D:\Geoton_2026\IMAGENES COMPLETAS\S2_AOI_UNIDA_2019_M6_10_DL.tif"
raster_2020 = r"D:\Geoton_2026\IMAGENES COMPLETAS\S2_AOI_UNIDA_2020_M6_10_DL.tif"
raster_2025 = r"D:\Geoton_2026\IMAGENES COMPLETAS\S2_AOI_UNIDA_2025_M6_10_DL.tif"

gpkg_2019 = r"D:\Geoton_2026\Ultimas_Capas\mineria_manual_2019.gpkg"
gpkg_2020 = r"D:\Geoton_2026\Ultimas_Capas\mineria_manual_2020.gpkg"
gpkg_2025 = r"D:\Geoton_2026\Ultimas_Capas\mineria_manual_2025.gpkg"

salida_dir = r"D:\Geoton_2026\IMAGENES COMPLETAS\CRS CORREGIDAS"

# Si tus GPKG tienen una sola capa, deja None.
# Si tienen varias capas, escribe el nombre de la capa.
layer_2019 = None
layer_2020 = None
layer_2025 = None


# ============================================================
# 2. FUNCIONES
# ============================================================

def leer_crs_raster(raster_path):
    with rasterio.open(raster_path) as src:
        return src.crs


def leer_info_raster(raster_path):
    with rasterio.open(raster_path) as src:
        info = {
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "transform": src.transform,
            "res": src.res,
            "count": src.count,
            "nodata": src.nodata
        }
    return info


def leer_gpkg(gpkg_path, layer=None):
    if layer is None:
        return gpd.read_file(gpkg_path)
    return gpd.read_file(gpkg_path, layer=layer)


def crear_bbox_raster(raster_path):
    with rasterio.open(raster_path) as src:
        b = src.bounds
        crs = src.crs

    bbox_geom = box(b.left, b.bottom, b.right, b.top)
    bbox_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bbox_geom], crs=crs)
    return bbox_gdf


def validar_interseccion(gdf, raster_path):
    bbox_raster = crear_bbox_raster(raster_path)

    if gdf.crs != bbox_raster.crs:
        gdf = gdf.to_crs(bbox_raster.crs)

    interseccion = gdf.intersects(bbox_raster.geometry.iloc[0])
    n_intersecta = interseccion.sum()

    return n_intersecta, len(gdf)


def validar_y_corregir_crs(raster_path, gpkg_path, output_gpkg_path, layer=None):
    print("\n====================================================")
    print(f"Raster: {raster_path}")
    print(f"GPKG:   {gpkg_path}")
    print("====================================================")

    raster_info = leer_info_raster(raster_path)
    raster_crs = raster_info["crs"]

    print(f"[RASTER] CRS: {raster_crs}")
    print(f"[RASTER] Tamaño: {raster_info['width']} x {raster_info['height']}")
    print(f"[RASTER] Resolución: {raster_info['res']}")
    print(f"[RASTER] Bandas: {raster_info['count']}")
    print(f"[RASTER] Bounds: {raster_info['bounds']}")

    gdf = leer_gpkg(gpkg_path, layer=layer)

    print(f"[GPKG] Polígonos originales: {len(gdf)}")
    print(f"[GPKG] CRS original: {gdf.crs}")

    if gdf.crs is None:
        raise ValueError(
            "El GPKG no tiene CRS definido. "
            "Debes asignar su CRS correcto en QGIS antes de reproyectar."
        )

    # Corregir geometrías inválidas si existen
    invalidas = (~gdf.is_valid).sum()
    if invalidas > 0:
        print(f"[ADVERTENCIA] Geometrías inválidas encontradas: {invalidas}")
        print("[INFO] Corrigiendo geometrías con buffer(0)...")
        gdf["geometry"] = gdf.geometry.buffer(0)

    # Reproyectar si CRS es diferente
    if gdf.crs != raster_crs:
        print(f"[INFO] CRS diferente. Reproyectando GPKG:")
        print(f"       De: {gdf.crs}")
        print(f"       A:  {raster_crs}")
        gdf_corr = gdf.to_crs(raster_crs)
    else:
        print("[OK] El GPKG ya está en el mismo CRS del raster.")
        gdf_corr = gdf.copy()

    # Validar intersección con raster
    n_intersecta, total = validar_interseccion(gdf_corr, raster_path)

    print(f"[VALIDACIÓN] Polígonos que intersectan el raster: {n_intersecta} / {total}")

    if n_intersecta == 0:
        print("[ERROR] Ningún polígono intersecta el raster.")
        print("Posibles causas:")
        print("1. El CRS original del GPKG está mal asignado.")
        print("2. El raster no corresponde al área del GPKG.")
        print("3. El GPKG está en otra zona UTM.")
        print("4. El raster fue recortado a otra área.")
        raise ValueError("No hay intersección espacial entre GPKG y raster.")

    # Guardar GPKG corregido
    output_gpkg_path = Path(output_gpkg_path)
    output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_corr.to_file(output_gpkg_path, driver="GPKG")

    print(f"[OK] GPKG corregido guardado en:")
    print(f"     {output_gpkg_path}")

    return gdf_corr


# ============================================================
# 3. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    salida_dir = Path(salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)

    salida_2019 = salida_dir / "mineria_manual_2019_reproyectado.gpkg"
    salida_2020 = salida_dir / "mineria_manual_2020_reproyectado.gpkg"
    salida_2025 = salida_dir / "mineria_manual_2025_reproyectado.gpkg"

    validar_y_corregir_crs(
        raster_path=raster_2019,
        gpkg_path=gpkg_2019,
        output_gpkg_path=salida_2019,
        layer=layer_2019
    )

    validar_y_corregir_crs(
        raster_path=raster_2020,
        gpkg_path=gpkg_2020,
        output_gpkg_path=salida_2020,
        layer=layer_2020
    )

    validar_y_corregir_crs(
        raster_path=raster_2025,
        gpkg_path=gpkg_2025,
        output_gpkg_path=salida_2025,
        layer=layer_2025
    )

    print("\n====================================================")
    print("[OK] VALIDACIÓN Y CORRECCIÓN DE CRS FINALIZADA")
    print("====================================================")
    