Mustatil Trainer - GeoPackage without GDAL/Fiona/GeoPandas

This version writes .gpkg directly with Python sqlite3.

No need to install:
- fiona
- geopandas
- GDAL
- pyogrio

GeoPackage export is built in and should open in QGIS.

Use:
1. Start with START_WITH_ONNX314_ENV.bat or START_WITH_ANACONDA.bat
2. In Detect Maps choose output:
   .gpkg
3. Run detection
4. Load the resulting .gpkg into QGIS

Layer name:
mustatil_detections

This avoids the Fiona error:
"A GDAL API version must be specified"
