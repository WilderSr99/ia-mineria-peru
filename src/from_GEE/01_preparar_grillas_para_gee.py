import geopandas as gpd
import pandas as pd
from pathlib import Path
import shutil
import zipfile

# ============================================================
# CONFIGURACIÓN
# ============================================================

entrada_gpkg = r"RUTA"

carpeta_base_salida = Path(
    r"RUTA"
)

nombre_capa_salida = "aoi_grillas_ordenadas"
nombre_shp = "aoi_grillas_gee"

tile_px = 1024
res_m = 10
lado_m = tile_px * res_m
lado_km = lado_m / 1000

# Para zona amazónica sur/centro oriental del Perú
crs_metrico = "EPSG:32719"

# ============================================================
# RUTAS DE SALIDA
# ============================================================

carpeta_base_salida.mkdir(parents=True, exist_ok=True)

salida_gpkg = carpeta_base_salida / "aoi_grillas_ordenadas.gpkg"
carpeta_shp = carpeta_base_salida / "aoi_grillas_shp"
salida_zip = carpeta_base_salida / "aoi_grillas_gee.zip"

# ============================================================
# 1. LEER GPKG
# ============================================================

gdf = gpd.read_file(entrada_gpkg)

print("==============================================")
print("LECTURA DEL GPKG")
print("CRS original:", gdf.crs)
print("Cantidad de grillas:", len(gdf))
print("Columnas encontradas:")
print(gdf.columns.tolist())
print("==============================================")

if gdf.crs is None:
    raise ValueError("El GPKG no tiene CRS definido. Define el CRS en QGIS antes de continuar.")

# ============================================================
# 2. LIMPIAR GEOMETRÍAS
# ============================================================

gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

gdf["geometry"] = gdf["geometry"].buffer(0)

gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

# ============================================================
# 3. CREAR O REPARAR ID_GRID
# ============================================================

if "id_grid" not in gdf.columns:
    gdf["id_grid"] = range(1, len(gdf) + 1)
else:
    ids_nuevos = pd.Series(
        range(1, len(gdf) + 1),
        index=gdf.index
    )
    gdf["id_grid"] = gdf["id_grid"].fillna(ids_nuevos)

gdf["id_grid"] = gdf["id_grid"].astype(int)

# ============================================================
# 4. ORDENAMIENTO ESPACIAL
# ============================================================
# Ordena las grillas:
# primero de norte a sur,
# luego de oeste a este.

gdf_wgs = gdf.to_crs(epsg=4326).copy()

# Usamos representative_point para evitar warnings de centroides en CRS geográfico
gdf_wgs["centro_x"] = gdf_wgs.geometry.representative_point().x
gdf_wgs["centro_y"] = gdf_wgs.geometry.representative_point().y

gdf_wgs = gdf_wgs.sort_values(
    by=["centro_y", "centro_x"],
    ascending=[False, True]
).reset_index(drop=True)

gdf_wgs["orden"] = gdf_wgs.index + 1

# ============================================================
# 5. REGENERAR FILA Y COLUMNA
# ============================================================
# Como ahora tienes 17 grillas, agrupamos filas según cercanía vertical.

tol_y = 0.02  # aprox. 2 km; si agrupa mal, subir a 0.03 o bajar a 0.01

filas = []
fila_actual = 0
ultimo_y = None

for y in gdf_wgs["centro_y"].values:
    if ultimo_y is None:
        fila_actual = 0
    else:
        if abs(y - ultimo_y) > tol_y:
            fila_actual += 1

    filas.append(fila_actual)
    ultimo_y = y

gdf_wgs["fila"] = filas

gdf_wgs["columna"] = (
    gdf_wgs
    .groupby("fila")["centro_x"]
    .rank(method="first", ascending=True)
    .astype(int) - 1
)

gdf_wgs = gdf_wgs.sort_values(
    by=["fila", "columna", "orden"],
    ascending=[True, True, True]
).reset_index(drop=True)

gdf_wgs["orden"] = gdf_wgs.index + 1
gdf_wgs["tile_ord"] = gdf_wgs["orden"].apply(lambda x: f"grid_{x:03d}")
gdf_wgs["tile_name"] = gdf_wgs["orden"].apply(lambda x: f"S2_grid_{x:03d}")

# ============================================================
# 6. COMPLETAR ATRIBUTOS
# ============================================================

gdf_wgs["tile_px"] = tile_px
gdf_wgs["res_m"] = res_m
gdf_wgs["lado_m"] = lado_m
gdf_wgs["lado_km"] = lado_km

# Calcular áreas en CRS métrico
gdf_m = gdf_wgs.to_crs(crs_metrico).copy()

gdf_wgs["area_aoi_m2"] = gdf_m.geometry.area
gdf_wgs["area_aoi_ha"] = gdf_wgs["area_aoi_m2"] / 10000
gdf_wgs["area_celda_m2"] = lado_m * lado_m
gdf_wgs["porc_cobertura_aoi"] = (
    gdf_wgs["area_aoi_m2"] / gdf_wgs["area_celda_m2"] * 100
)

campos_redondear = [
    "lado_m",
    "lado_km",
    "area_aoi_m2",
    "area_aoi_ha",
    "area_celda_m2",
    "porc_cobertura_aoi"
]

for campo in campos_redondear:
    gdf_wgs[campo] = gdf_wgs[campo].round(4)

# ============================================================
# 7. REORDENAR COLUMNAS
# ============================================================

columnas_finales = [
    "orden",
    "id_grid",
    "fila",
    "columna",
    "tile_px",
    "res_m",
    "lado_m",
    "lado_km",
    "area_aoi_m2",
    "area_celda_m2",
    "area_aoi_ha",
    "porc_cobertura_aoi",
    "tile_name",
    "tile_ord",
    "geometry"
]

columnas_existentes = [c for c in columnas_finales if c in gdf_wgs.columns]
gdf_wgs = gdf_wgs[columnas_existentes]

# ============================================================
# 8. GUARDAR GPKG ORDENADO
# ============================================================

if salida_gpkg.exists():
    salida_gpkg.unlink()

gdf_wgs.to_file(
    salida_gpkg,
    layer=nombre_capa_salida,
    driver="GPKG"
)

print("==============================================")
print("GPKG ordenado generado correctamente")
print("Ruta:", salida_gpkg)
print("Cantidad de grillas:", len(gdf_wgs))
print(gdf_wgs[["orden", "id_grid", "fila", "columna", "tile_name", "tile_ord"]])
print("==============================================")

# ============================================================
# 9. CONVERTIR A SHAPEFILE PARA GEE
# ============================================================

if carpeta_shp.exists():
    shutil.rmtree(carpeta_shp)

carpeta_shp.mkdir(parents=True, exist_ok=True)

gdf_shp = gdf_wgs.to_crs(epsg=4326).copy()

# Shapefile permite nombres de máximo 10 caracteres
renombrar = {
    "area_aoi_m2": "area_m2",
    "area_celda_m2": "celda_m2",
    "area_aoi_ha": "area_ha",
    "porc_cobertura_aoi": "cobert"
}

gdf_shp = gdf_shp.rename(columns=renombrar)

campos_shp = [
    "orden",
    "id_grid",
    "fila",
    "columna",
    "tile_px",
    "res_m",
    "lado_m",
    "lado_km",
    "area_m2",
    "celda_m2",
    "area_ha",
    "cobert",
    "tile_name",
    "tile_ord",
    "geometry"
]

campos_existentes = [c for c in campos_shp if c in gdf_shp.columns]
gdf_shp = gdf_shp[campos_existentes]

ruta_shp = carpeta_shp / f"{nombre_shp}.shp"

gdf_shp.to_file(
    ruta_shp,
    driver="ESRI Shapefile",
    encoding="UTF-8"
)

print("Shapefile generado:", ruta_shp)

# ============================================================
# 10. COMPRIMIR SHAPEFILE EN ZIP
# ============================================================

if salida_zip.exists():
    salida_zip.unlink()

extensiones_necesarias = [".shp", ".shx", ".dbf", ".prj", ".cpg"]

with zipfile.ZipFile(salida_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for archivo in carpeta_shp.iterdir():
        if archivo.suffix.lower() in extensiones_necesarias:
            zipf.write(archivo, arcname=archivo.name)

print("==============================================")
print("ZIP generado correctamente para subir a GEE")
print("Ruta:", salida_zip)
print("Archivos incluidos:")
with zipfile.ZipFile(salida_zip, "r") as zipf:
    for nombre in zipf.namelist():
        print(" -", nombre)
print("==============================================")