# ============================================================
# CRUCE TERRITORIAL DE MINERÍA ILEGAL 2019-2025
# Con ríos navegables, comunidades nativas, ANP y zonas de amortiguamiento
# Autor: Wilder Teddy Sebastian Rios
# ============================================================

import os
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.ops import unary_union

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================

# Carpeta donde están los resultados del análisis temporal
CARPETA_TEMPORAL = r"D:\Geoton_2026\PREDICCIONES MINERIA\AREA DE ESTUDIO\CAPAS_PRED\capas_consolidado\analisis_temporal_resultados"

# Carpeta donde guardarás los cruces territoriales
CARPETA_SALIDA = os.path.join(CARPETA_TEMPORAL, "cruce_territorial")

os.makedirs(CARPETA_SALIDA, exist_ok=True)

# CRS de trabajo
# Según tu mapa: UTM Zone 19S
# EPSG:32719 = WGS 84 / UTM zone 19S
CRS_TRABAJO = "EPSG:32719"


# ============================================================
# 2. RUTAS DE CAPAS TERRITORIALES
# MODIFICA ESTAS RUTAS SEGÚN TU CARPETA REAL
# ============================================================

RUTA_RIOS = r"C:\Users\decg112\Downloads\GEOTON DATOS\Capas\Rios\Rios_Navegables_INDECI_geogpsperu_SuyoPomalia\Rios_Navegables_INDECI_geogpsperu_SuyoPomalia.shp"
RUTA_COMUNIDADES = r"C:\Users\decg112\Downloads\GEOTON DATOS\AREAS DE CONSERVACIÓN\comunidad-nativa-titulada\comunidad-nativa-titulada.shp"
RUTA_ANP = r"C:\Users\decg112\Downloads\GEOTON DATOS\AREAS DE CONSERVACIÓN\area-natural-protegida\area-natural-protegida.shp"
RUTA_ZA = r"C:\Users\decg112\Downloads\GEOTON DATOS\AREAS DE CONSERVACIÓN\Zonas de Amortiguamiento\ZonasdeAmortiguamiento.shp"


# ============================================================
# 3. CAMPOS PRINCIPALES SEGÚN TUS TABLAS DE ATRIBUTOS
# ============================================================

CAPAS_TERRITORIALES = {
    "rios_navegables": {
        "ruta": RUTA_RIOS,
        "campo_id": "objectid",
        "campo_nombre": "nombre",
        "tipo": "rio"
    },
    "comunidades_nativas": {
        "ruta": RUTA_COMUNIDADES,
        "campo_id": "feature_id",
        "campo_nombre": "nombre",
        "tipo": "territorio"
    },
    "areas_naturales_protegidas": {
        "ruta": RUTA_ANP,
        "campo_id": "feature_id",
        "campo_nombre": "nombre",
        "tipo": "territorio"
    },
    "zonas_amortiguamiento": {
        "ruta": RUTA_ZA,
        "campo_id": "objectid_1",
        "campo_nombre": "anp_nomb",
        "tipo": "territorio"
    }
}


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def reparar_geometrias(gdf):
    """
    Repara geometrías inválidas y elimina geometrías nulas o vacías.
    """
    gdf = gdf.copy()

    if len(gdf) == 0:
        return gdf

    gdf = gdf[~gdf.geometry.isna()]

    if len(gdf) == 0:
        return gdf

    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty]

    return gdf


def leer_capa_vectorial(ruta, crs_trabajo=CRS_TRABAJO):
    """
    Lee una capa vectorial y la reproyecta al CRS de trabajo.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe la capa: {ruta}")

    gdf = gpd.read_file(ruta)

    if gdf.crs is None:
        print(f"Advertencia: la capa {ruta} no tiene CRS. Se asignará {crs_trabajo}.")
        gdf = gdf.set_crs(crs_trabajo)
    else:
        gdf = gdf.to_crs(crs_trabajo)

    gdf = reparar_geometrias(gdf)

    return gdf


def asegurar_campos(gdf, campo_id, campo_nombre, nombre_capa):
    """
    Verifica si existen los campos indicados.
    Si no existen, crea campos alternativos.
    """
    gdf = gdf.copy()

    if campo_id not in gdf.columns:
        print(f"Advertencia: campo_id '{campo_id}' no existe en {nombre_capa}. Se usará índice.")
        gdf[campo_id] = range(1, len(gdf) + 1)

    if campo_nombre not in gdf.columns:
        print(f"Advertencia: campo_nombre '{campo_nombre}' no existe en {nombre_capa}. Se usará nombre genérico.")
        gdf[campo_nombre] = nombre_capa + "_" + gdf[campo_id].astype(str)

    return gdf


def cargar_mineria_anual(anio):
    """
    Carga la minería disuelta anual generada por el análisis temporal.
    """
    ruta = os.path.join(CARPETA_TEMPORAL, f"mineria_disuelta_{anio}.gpkg")

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe la capa anual: {ruta}")

    gdf = gpd.read_file(ruta)

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_TRABAJO)
    else:
        gdf = gdf.to_crs(CRS_TRABAJO)

    gdf = reparar_geometrias(gdf)

    gdf["anio"] = anio
    gdf["tipo_mineria"] = f"mineria_{anio}"

    return gdf


def cargar_capa_temporal(nombre_archivo, tipo_mineria):
    """
    Carga una capa temporal específica: acumulado o cambio total.
    """
    ruta = os.path.join(CARPETA_TEMPORAL, nombre_archivo)

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe la capa temporal: {ruta}")

    gdf = gpd.read_file(ruta)

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_TRABAJO)
    else:
        gdf = gdf.to_crs(CRS_TRABAJO)

    gdf = reparar_geometrias(gdf)
    gdf["tipo_mineria"] = tipo_mineria

    return gdf


def explotar_multipart(gdf):
    """
    Separa geometrías multipart en partes individuales.
    """
    gdf = gdf.copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf


def calcular_interseccion(mineria_gdf, capa_gdf, nombre_capa, campo_id, campo_nombre):
    """
    Calcula intersección espacial entre minería y una capa territorial.
    Devuelve capa de intersecciones y tabla resumen.
    """
    mineria_gdf = mineria_gdf.copy()
    capa_gdf = capa_gdf.copy()

    mineria_gdf = reparar_geometrias(mineria_gdf)
    capa_gdf = reparar_geometrias(capa_gdf)

    if len(mineria_gdf) == 0 or len(capa_gdf) == 0:
        return None, pd.DataFrame()

    # Reducir columnas territoriales
    columnas_base = [campo_id, campo_nombre, "geometry"]
    columnas_existentes = [c for c in columnas_base if c in capa_gdf.columns]
    capa_reducida = capa_gdf[columnas_existentes].copy()

    capa_reducida = capa_reducida.rename(
        columns={
            campo_id: "id_unidad",
            campo_nombre: "nombre_unidad"
        }
    )

    # Asegurar columnas de minería
    if "anio" not in mineria_gdf.columns:
        mineria_gdf["anio"] = None

    if "tipo_mineria" not in mineria_gdf.columns:
        mineria_gdf["tipo_mineria"] = "mineria"

    if "tipo_cambio" not in mineria_gdf.columns:
        mineria_gdf["tipo_cambio"] = mineria_gdf["tipo_mineria"]

    mineria_reducida = mineria_gdf[
        ["anio", "tipo_mineria", "tipo_cambio", "geometry"]
    ].copy()

    # Intersección
    inter = gpd.overlay(
        mineria_reducida,
        capa_reducida,
        how="intersection",
        keep_geom_type=False
    )

    inter = reparar_geometrias(inter)

    if len(inter) == 0:
        return None, pd.DataFrame()

    inter["capa_cruce"] = nombre_capa
    inter["area_m2"] = inter.geometry.area
    inter["area_ha"] = inter["area_m2"] / 10000
    inter["area_km2"] = inter["area_ha"] / 100

    # Resumen por unidad
    resumen = (
        inter
        .groupby(
            [
                "capa_cruce",
                "id_unidad",
                "nombre_unidad",
                "anio",
                "tipo_mineria",
                "tipo_cambio"
            ],
            dropna=False
        )
        .agg(
            area_intersectada_ha=("area_ha", "sum"),
            area_intersectada_km2=("area_km2", "sum"),
            cantidad_fragmentos=("area_ha", "count")
        )
        .reset_index()
        .sort_values("area_intersectada_ha", ascending=False)
    )

    return inter, resumen


def calcular_resumen_total_por_capa(inter_gdf, nombre_capa):
    """
    Resume el área total intersectada por tipo de minería/cambio.
    """
    if inter_gdf is None or len(inter_gdf) == 0:
        return pd.DataFrame()

    resumen = (
        inter_gdf
        .groupby(
            ["capa_cruce", "anio", "tipo_mineria", "tipo_cambio"],
            dropna=False
        )
        .agg(
            area_total_intersectada_ha=("area_ha", "sum"),
            area_total_intersectada_km2=("area_km2", "sum"),
            cantidad_fragmentos=("area_ha", "count")
        )
        .reset_index()
    )

    resumen["capa_cruce"] = nombre_capa

    return resumen


def crear_buffer_rios(gdf_rios, distancia_m):
    """
    Crea buffer alrededor de ríos.
    Sirve para analizar presión minera cercana a ríos.
    """
    gdf_buffer = gdf_rios.copy()
    gdf_buffer["geometry"] = gdf_buffer.geometry.buffer(distancia_m)
    gdf_buffer = reparar_geometrias(gdf_buffer)
    gdf_buffer["buffer_m"] = distancia_m

    return gdf_buffer


def guardar_gpkg(gdf, ruta):
    """
    Guarda una capa GPKG si no está vacía.
    """
    if gdf is not None and len(gdf) > 0:
        gdf.to_file(ruta, driver="GPKG")
        print(f"Capa guardada: {ruta}")
    else:
        print(f"No se guardó capa vacía: {ruta}")


# ============================================================
# 5. CARGAR CAPAS TERRITORIALES
# ============================================================

print("\n============================================================")
print("CARGANDO CAPAS TERRITORIALES")
print("============================================================")

capas_auxiliares = {}

for nombre_capa, cfg in CAPAS_TERRITORIALES.items():
    print(f"\nCargando: {nombre_capa}")
    gdf = leer_capa_vectorial(cfg["ruta"])
    gdf = asegurar_campos(
        gdf,
        cfg["campo_id"],
        cfg["campo_nombre"],
        nombre_capa
    )

    capas_auxiliares[nombre_capa] = gdf

    print(f"Objetos cargados: {len(gdf)}")
    print(f"CRS: {gdf.crs}")


# ============================================================
# 6. CARGAR CAPAS DE MINERÍA
# ============================================================

print("\n============================================================")
print("CARGANDO CAPAS TEMPORALES DE MINERÍA")
print("============================================================")

anios = list(range(2019, 2026))

capas_mineria = {}

# Minería anual
for anio in anios:
    print(f"Cargando minería anual {anio}")
    capas_mineria[f"mineria_{anio}"] = cargar_mineria_anual(anio)

# Cambio total 2019-2025
print("Cargando cambio total 2019-2025")
capas_mineria["cambio_total_2019_2025"] = cargar_capa_temporal(
    "cambio_total_mineria_2019_2025.gpkg",
    "cambio_total_2019_2025"
)

# Acumulado multitemporal
print("Cargando acumulado multitemporal 2019-2025")
capas_mineria["acumulado_2019_2025"] = cargar_capa_temporal(
    "mineria_acumulada_2019_2025.gpkg",
    "acumulado_2019_2025"
)


# ============================================================
# 7. CRUCE DE MINERÍA ANUAL CON CAPAS TERRITORIALES
# ============================================================

print("\n============================================================")
print("CRUCE DE MINERÍA ANUAL CON CAPAS TERRITORIALES")
print("============================================================")

resumenes_anuales = []
resumenes_totales_anuales = []

for nombre_mineria, gdf_mineria in capas_mineria.items():

    # Solo procesar minería anual en este bloque
    if not nombre_mineria.startswith("mineria_"):
        continue

    print(f"\nProcesando: {nombre_mineria}")

    for nombre_capa, cfg in CAPAS_TERRITORIALES.items():
        print(f"  Cruzando con: {nombre_capa}")

        gdf_aux = capas_auxiliares[nombre_capa]

        inter, resumen = calcular_interseccion(
            gdf_mineria,
            gdf_aux,
            nombre_capa,
            cfg["campo_id"],
            cfg["campo_nombre"]
        )

        if resumen is not None and len(resumen) > 0:
            resumenes_anuales.append(resumen)

        resumen_total = calcular_resumen_total_por_capa(inter, nombre_capa)

        if resumen_total is not None and len(resumen_total) > 0:
            resumenes_totales_anuales.append(resumen_total)

        salida_inter = os.path.join(
            CARPETA_SALIDA,
            f"interseccion_{nombre_mineria}_{nombre_capa}.gpkg"
        )

        guardar_gpkg(inter, salida_inter)


if len(resumenes_anuales) > 0:
    df_resumen_anual_unidad = pd.concat(resumenes_anuales, ignore_index=True)
else:
    df_resumen_anual_unidad = pd.DataFrame()

if len(resumenes_totales_anuales) > 0:
    df_resumen_anual_total = pd.concat(resumenes_totales_anuales, ignore_index=True)
else:
    df_resumen_anual_total = pd.DataFrame()


# ============================================================
# 8. CRUCE DEL CAMBIO TOTAL 2019-2025
# ============================================================

print("\n============================================================")
print("CRUCE DEL CAMBIO TOTAL 2019-2025")
print("============================================================")

resumenes_cambio_total = []
resumenes_cambio_total_por_capa = []

gdf_cambio_total = capas_mineria["cambio_total_2019_2025"]

for nombre_capa, cfg in CAPAS_TERRITORIALES.items():
    print(f"\nCruzando cambio total con: {nombre_capa}")

    gdf_aux = capas_auxiliares[nombre_capa]

    inter, resumen = calcular_interseccion(
        gdf_cambio_total,
        gdf_aux,
        nombre_capa,
        cfg["campo_id"],
        cfg["campo_nombre"]
    )

    if resumen is not None and len(resumen) > 0:
        resumenes_cambio_total.append(resumen)

    resumen_total = calcular_resumen_total_por_capa(inter, nombre_capa)

    if resumen_total is not None and len(resumen_total) > 0:
        resumenes_cambio_total_por_capa.append(resumen_total)

    salida_inter = os.path.join(
        CARPETA_SALIDA,
        f"interseccion_cambio_total_2019_2025_{nombre_capa}.gpkg"
    )

    guardar_gpkg(inter, salida_inter)


if len(resumenes_cambio_total) > 0:
    df_resumen_cambio_total_unidad = pd.concat(
        resumenes_cambio_total,
        ignore_index=True
    )
else:
    df_resumen_cambio_total_unidad = pd.DataFrame()

if len(resumenes_cambio_total_por_capa) > 0:
    df_resumen_cambio_total_capa = pd.concat(
        resumenes_cambio_total_por_capa,
        ignore_index=True
    )
else:
    df_resumen_cambio_total_capa = pd.DataFrame()


# ============================================================
# 9. CRUCE DEL ACUMULADO MULTITEMPORAL
# ============================================================

print("\n============================================================")
print("CRUCE DEL ACUMULADO MULTITEMPORAL 2019-2025")
print("============================================================")

resumenes_acumulado = []
resumenes_acumulado_por_capa = []

gdf_acumulado = capas_mineria["acumulado_2019_2025"]

for nombre_capa, cfg in CAPAS_TERRITORIALES.items():
    print(f"\nCruzando acumulado con: {nombre_capa}")

    gdf_aux = capas_auxiliares[nombre_capa]

    inter, resumen = calcular_interseccion(
        gdf_acumulado,
        gdf_aux,
        nombre_capa,
        cfg["campo_id"],
        cfg["campo_nombre"]
    )

    if resumen is not None and len(resumen) > 0:
        resumenes_acumulado.append(resumen)

    resumen_total = calcular_resumen_total_por_capa(inter, nombre_capa)

    if resumen_total is not None and len(resumen_total) > 0:
        resumenes_acumulado_por_capa.append(resumen_total)

    salida_inter = os.path.join(
        CARPETA_SALIDA,
        f"interseccion_acumulado_2019_2025_{nombre_capa}.gpkg"
    )

    guardar_gpkg(inter, salida_inter)


if len(resumenes_acumulado) > 0:
    df_resumen_acumulado_unidad = pd.concat(
        resumenes_acumulado,
        ignore_index=True
    )
else:
    df_resumen_acumulado_unidad = pd.DataFrame()

if len(resumenes_acumulado_por_capa) > 0:
    df_resumen_acumulado_capa = pd.concat(
        resumenes_acumulado_por_capa,
        ignore_index=True
    )
else:
    df_resumen_acumulado_capa = pd.DataFrame()


# ============================================================
# 10. ANÁLISIS DE PROXIMIDAD A RÍOS
# BUFFERS DE 100 M, 500 M Y 1000 M
# ============================================================

print("\n============================================================")
print("ANÁLISIS DE PROXIMIDAD A RÍOS")
print("============================================================")

DISTANCIAS_BUFFER_RIOS = [100, 500, 1000]

gdf_rios = capas_auxiliares["rios_navegables"]
cfg_rios = CAPAS_TERRITORIALES["rios_navegables"]

resumenes_buffers_rios = []

for distancia in DISTANCIAS_BUFFER_RIOS:
    print(f"\nCreando buffer de ríos: {distancia} m")

    gdf_buffer = crear_buffer_rios(gdf_rios, distancia)

    salida_buffer = os.path.join(
        CARPETA_SALIDA,
        f"buffer_rios_{distancia}m.gpkg"
    )

    guardar_gpkg(gdf_buffer, salida_buffer)

    # Cruzar minería anual con buffer
    for anio in anios:
        nombre_mineria = f"mineria_{anio}"
        gdf_mineria = capas_mineria[nombre_mineria]

        inter, resumen = calcular_interseccion(
            gdf_mineria,
            gdf_buffer,
            f"buffer_rios_{distancia}m",
            cfg_rios["campo_id"],
            cfg_rios["campo_nombre"]
        )

        if resumen is not None and len(resumen) > 0:
            resumen["buffer_m"] = distancia
            resumenes_buffers_rios.append(resumen)

        salida_inter = os.path.join(
            CARPETA_SALIDA,
            f"interseccion_mineria_{anio}_buffer_rios_{distancia}m.gpkg"
        )

        guardar_gpkg(inter, salida_inter)

    # Cruzar cambio total 2019-2025 con buffer
    inter_cambio, resumen_cambio = calcular_interseccion(
        gdf_cambio_total,
        gdf_buffer,
        f"buffer_rios_{distancia}m",
        cfg_rios["campo_id"],
        cfg_rios["campo_nombre"]
    )

    if resumen_cambio is not None and len(resumen_cambio) > 0:
        resumen_cambio["buffer_m"] = distancia
        resumenes_buffers_rios.append(resumen_cambio)

    salida_inter_cambio = os.path.join(
        CARPETA_SALIDA,
        f"interseccion_cambio_total_2019_2025_buffer_rios_{distancia}m.gpkg"
    )

    guardar_gpkg(inter_cambio, salida_inter_cambio)


if len(resumenes_buffers_rios) > 0:
    df_resumen_buffers_rios = pd.concat(
        resumenes_buffers_rios,
        ignore_index=True
    )
else:
    df_resumen_buffers_rios = pd.DataFrame()


# ============================================================
# 11. TABLAS PRINCIPALES PARA INFORME
# ============================================================

print("\n============================================================")
print("GENERANDO TABLAS CONSOLIDADAS")
print("============================================================")

# 11.1 Tabla de presión anual por capa
if len(df_resumen_anual_total) > 0:
    tabla_presion_anual = (
        df_resumen_anual_total
        .groupby(["capa_cruce", "anio"], dropna=False)
        .agg(
            area_intersectada_ha=("area_total_intersectada_ha", "sum"),
            area_intersectada_km2=("area_total_intersectada_km2", "sum")
        )
        .reset_index()
        .sort_values(["capa_cruce", "anio"])
    )
else:
    tabla_presion_anual = pd.DataFrame()


# 11.2 Tabla de cambio total por capa
if len(df_resumen_cambio_total_capa) > 0:
    tabla_cambio_total = (
        df_resumen_cambio_total_capa
        .groupby(["capa_cruce", "tipo_cambio"], dropna=False)
        .agg(
            area_intersectada_ha=("area_total_intersectada_ha", "sum"),
            area_intersectada_km2=("area_total_intersectada_km2", "sum")
        )
        .reset_index()
        .sort_values(["capa_cruce", "tipo_cambio"])
    )
else:
    tabla_cambio_total = pd.DataFrame()


# 11.3 Ranking de unidades más afectadas por expansión nueva 2019-2025
if len(df_resumen_cambio_total_unidad) > 0:
    ranking_expansion_unidades = df_resumen_cambio_total_unidad[
        df_resumen_cambio_total_unidad["tipo_cambio"] == "expansion_nueva_2019_2025"
    ].copy()

    ranking_expansion_unidades = ranking_expansion_unidades.sort_values(
        "area_intersectada_ha",
        ascending=False
    )
else:
    ranking_expansion_unidades = pd.DataFrame()


# 11.4 Tabla de acumulado por capa
if len(df_resumen_acumulado_capa) > 0:
    tabla_acumulado = (
        df_resumen_acumulado_capa
        .groupby(["capa_cruce"], dropna=False)
        .agg(
            area_acumulada_intersectada_ha=("area_total_intersectada_ha", "sum"),
            area_acumulada_intersectada_km2=("area_total_intersectada_km2", "sum")
        )
        .reset_index()
        .sort_values("area_acumulada_intersectada_ha", ascending=False)
    )
else:
    tabla_acumulado = pd.DataFrame()


# ============================================================
# 12. EXPORTAR TABLAS A CSV Y EXCEL
# ============================================================

print("\n============================================================")
print("EXPORTANDO TABLAS")
print("============================================================")

# CSV
rutas_csv = {
    "resumen_anual_por_unidad.csv": df_resumen_anual_unidad,
    "resumen_anual_por_capa.csv": tabla_presion_anual,
    "resumen_cambio_total_por_unidad.csv": df_resumen_cambio_total_unidad,
    "resumen_cambio_total_por_capa.csv": tabla_cambio_total,
    "ranking_expansion_nueva_2019_2025.csv": ranking_expansion_unidades,
    "resumen_acumulado_por_unidad.csv": df_resumen_acumulado_unidad,
    "resumen_acumulado_por_capa.csv": tabla_acumulado,
    "resumen_buffers_rios.csv": df_resumen_buffers_rios
}

for nombre_csv, df in rutas_csv.items():
    ruta_csv = os.path.join(CARPETA_SALIDA, nombre_csv)

    if df is not None and len(df) > 0:
        df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
        print(f"CSV guardado: {ruta_csv}")
    else:
        print(f"No se exportó CSV vacío: {nombre_csv}")


# Excel
ruta_excel = os.path.join(
    CARPETA_SALIDA,
    "resumen_cruce_territorial_mineria_2019_2025.xlsx"
)

try:
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:

        if len(df_resumen_anual_unidad) > 0:
            df_resumen_anual_unidad.to_excel(
                writer,
                sheet_name="anual_por_unidad",
                index=False
            )

        if len(tabla_presion_anual) > 0:
            tabla_presion_anual.to_excel(
                writer,
                sheet_name="anual_por_capa",
                index=False
            )

        if len(df_resumen_cambio_total_unidad) > 0:
            df_resumen_cambio_total_unidad.to_excel(
                writer,
                sheet_name="cambio_total_unidad",
                index=False
            )

        if len(tabla_cambio_total) > 0:
            tabla_cambio_total.to_excel(
                writer,
                sheet_name="cambio_total_capa",
                index=False
            )

        if len(ranking_expansion_unidades) > 0:
            ranking_expansion_unidades.to_excel(
                writer,
                sheet_name="ranking_expansion",
                index=False
            )

        if len(df_resumen_acumulado_unidad) > 0:
            df_resumen_acumulado_unidad.to_excel(
                writer,
                sheet_name="acumulado_unidad",
                index=False
            )

        if len(tabla_acumulado) > 0:
            tabla_acumulado.to_excel(
                writer,
                sheet_name="acumulado_capa",
                index=False
            )

        if len(df_resumen_buffers_rios) > 0:
            df_resumen_buffers_rios.to_excel(
                writer,
                sheet_name="buffers_rios",
                index=False
            )

    print(f"Excel guardado: {ruta_excel}")

except ModuleNotFoundError:
    print("No se encontró openpyxl. Se exportaron los resultados en CSV.")
    print("Para exportar Excel instala: conda install -c conda-forge openpyxl")


# ============================================================
# 13. GRÁFICOS PARA EL INFORME
# ============================================================

print("\n============================================================")
print("GENERANDO GRÁFICOS")
print("============================================================")

# Gráfico 1: presión anual por capa territorial
if len(tabla_presion_anual) > 0:
    for capa in tabla_presion_anual["capa_cruce"].unique():
        df_plot = tabla_presion_anual[
            tabla_presion_anual["capa_cruce"] == capa
        ].copy()

        plt.figure(figsize=(10, 6))
        plt.plot(
            df_plot["anio"],
            df_plot["area_intersectada_ha"],
            marker="o"
        )

        plt.title(f"Minería intersectada con {capa}, 2019-2025")
        plt.xlabel("Año")
        plt.ylabel("Área intersectada (ha)")
        plt.grid(True, alpha=0.3)

        for x, y in zip(df_plot["anio"], df_plot["area_intersectada_ha"]):
            plt.text(
                x,
                y,
                f"{y:,.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        plt.tight_layout()

        ruta_grafico = os.path.join(
            CARPETA_SALIDA,
            f"grafico_presion_anual_{capa}.png"
        )

        plt.savefig(ruta_grafico, dpi=300)
        plt.close()

        print(f"Gráfico guardado: {ruta_grafico}")


# Gráfico 2: cambio total por capa y tipo de cambio
if len(tabla_cambio_total) > 0:
    for capa in tabla_cambio_total["capa_cruce"].unique():
        df_plot = tabla_cambio_total[
            tabla_cambio_total["capa_cruce"] == capa
        ].copy()

        plt.figure(figsize=(10, 6))
        plt.bar(
            df_plot["tipo_cambio"],
            df_plot["area_intersectada_ha"]
        )

        plt.title(f"Cambio total 2019-2025 intersectado con {capa}")
        plt.xlabel("Tipo de cambio")
        plt.ylabel("Área intersectada (ha)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)

        for x, y in zip(df_plot["tipo_cambio"], df_plot["area_intersectada_ha"]):
            plt.text(
                x,
                y,
                f"{y:,.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        plt.tight_layout()

        ruta_grafico = os.path.join(
            CARPETA_SALIDA,
            f"grafico_cambio_total_{capa}.png"
        )

        plt.savefig(ruta_grafico, dpi=300)
        plt.close()

        print(f"Gráfico guardado: {ruta_grafico}")


# ============================================================
# 14. RESUMEN FINAL EN CONSOLA
# ============================================================

print("\n============================================================")
print("RESUMEN FINAL DEL CRUCE TERRITORIAL")
print("============================================================")

if len(tabla_cambio_total) > 0:
    print("\nCambio total 2019-2025 por capa territorial:")
    print(tabla_cambio_total)

if len(ranking_expansion_unidades) > 0:
    print("\nTop 15 unidades con mayor expansión nueva 2019-2025:")
    columnas_ranking = [
        "capa_cruce",
        "nombre_unidad",
        "tipo_cambio",
        "area_intersectada_ha",
        "area_intersectada_km2"
    ]

    columnas_existentes = [
        c for c in columnas_ranking
        if c in ranking_expansion_unidades.columns
    ]

    print(ranking_expansion_unidades[columnas_existentes].head(15))

if len(tabla_acumulado) > 0:
    print("\nMinería acumulada 2019-2025 por capa:")
    print(tabla_acumulado)

print("\nResultados guardados en:")
print(CARPETA_SALIDA)

print("============================================================")
print("Proceso finalizado correctamente.")
print("============================================================")