// ============================================================
// EXTRACCIÓN SENTINEL-2 POR GRILLAS 1024 x 1024
// AOI desde shapefile subido como asset
// ============================================================

// ------------------------------------------------------------
// 1. CONFIGURACIÓN
// ------------------------------------------------------------

// Cambia esta ruta por tu asset real



// Año o periodo de extracción
var anio = 2025;

// Puedes cambiar el periodo
var fechaInicio = anio + "-01-01";
var fechaFin = anio + "-12-31";

// Porcentaje máximo de nubosidad por imagen
var nubeMax = 40;

// Carpeta en Google Drive
var carpetaDrive = "S2_GRILLAS_GEE_" + anio;

// Escala de exportación Sentinel-2
var escala = 10;

// CRS recomendado para tu zona amazónica.
// Si tu AOI está entre -72 y -66, usa EPSG:32719.
// Si estuviera más al oeste, podrías usar EPSG:32718.
var crsSalida = "EPSG:32719";

// ------------------------------------------------------------
// 2. VERIFICAR GRILLAS
// ------------------------------------------------------------

print("Cantidad de grillas:", grillas.size());
print("Grillas:", grillas);

Map.centerObject(grillas, 11);
Map.addLayer(grillas, {color: "yellow"}, "Grillas AOI");

// ------------------------------------------------------------
// 3. FUNCIÓN DE MÁSCARA DE NUBES SENTINEL-2 SR
// ------------------------------------------------------------

function maskS2sr(image) {
  var scl = image.select("SCL");

  // Se eliminan:
  // 3 = sombra de nube
  // 8 = nube media probabilidad
  // 9 = nube alta probabilidad
  // 10 = cirros
  // 11 = nieve/hielo
  var mask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image.updateMask(mask);
}

// ------------------------------------------------------------
// 4. FUNCIÓN PARA ESCALAR BANDAS ÓPTICAS
// ------------------------------------------------------------

function scaleS2(image) {
  var bandas = image.select(["B2", "B3", "B4", "B8", "B11", "B12"])
    .multiply(0.0001);

  return image.addBands(bandas, null, true);
}

// ------------------------------------------------------------
// 5. FUNCIÓN PARA ÍNDICES ESPECTRALES
// ------------------------------------------------------------

function addIndices(image) {
  var blue = image.select("B2");
  var green = image.select("B3");
  var red = image.select("B4");
  var nir = image.select("B8");
  var swir1 = image.select("B11");
  var swir2 = image.select("B12");

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI");
  var ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI");
  var mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI");
  var nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR");
  var ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI");

  var bsi = swir1.add(red)
    .subtract(nir.add(blue))
    .divide(swir1.add(red).add(nir).add(blue))
    .rename("BSI");

  return image.addBands([
    ndvi,
    ndwi,
    mndwi,
    nbr,
    ndbi,
    bsi
  ]);
}

// ------------------------------------------------------------
// 6. COLECCIÓN SENTINEL-2
// ------------------------------------------------------------

var coleccion = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(grillas)
  .filterDate(fechaInicio, fechaFin)
  .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", nubeMax))
  .map(maskS2sr)
  .map(scaleS2)
  .map(addIndices);

print("Cantidad de imágenes Sentinel-2:", coleccion.size());
print("Colección Sentinel-2:", coleccion);

// ------------------------------------------------------------
// 7. COMPOSICIÓN MEDIANA
// ------------------------------------------------------------

var bandasSalida = [
  "B2", "B3", "B4", "B8", "B11", "B12",
  "NDVI", "NDWI", "MNDWI", "NBR", "NDBI", "BSI"
];

var composite = coleccion
  .median()
  .select(bandasSalida)
  .clip(grillas);

print("Composite 12 bandas:", composite);

Map.addLayer(
  composite,
  {bands: ["B4", "B3", "B2"], min: 0, max: 0.3},
  "Sentinel-2 RGB " + anio
);

// ------------------------------------------------------------
// 8. EXPORTAR UNA IMAGEN POR CADA GRILLA
// ------------------------------------------------------------

// Convertir FeatureCollection a lista
var listaGrillas = grillas.sort("orden").toList(grillas.size());

var n = grillas.size().getInfo();

print("Total de exportaciones a crear:", n);

for (var i = 0; i < n; i++) {
  var feature = ee.Feature(listaGrillas.get(i));
  var geom = feature.geometry();

  var orden = feature.get("orden").getInfo();
  var tileOrd = feature.get("tile_ord").getInfo();

  // Si tile_ord viniera vacío, crear nombre manual
  if (tileOrd === null || tileOrd === undefined) {
    tileOrd = "grid_" + String(orden).padStart(3, "0");
  }

  var nombre = "S2_" + anio + "_" + tileOrd + "_12bandas";

  Export.image.toDrive({
    image: composite.clip(geom),
    description: nombre,
    folder: carpetaDrive,
    fileNamePrefix: nombre,
    region: geom,
    scale: escala,
    crs: crsSalida,
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"
  });
}
