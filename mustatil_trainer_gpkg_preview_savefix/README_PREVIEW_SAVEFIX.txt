Mustatil Trainer - GeoPackage Preview Save Fix

Fixes the case where detection reaches the last tile but then appears to stop at:
    Writing GeoPackage output...

Changes:
1. .preview.geojson is written BEFORE .gpkg
2. .gpkg is created locally in the Windows temp folder
3. finished .gpkg is copied to the final target path
4. if QGIS locks the old .gpkg, a new timestamped file is created:
       output_new_123456789.gpkg
5. preview reload uses the sidecar:
       output.preview.geojson

This avoids SQLite locks and network-drive hangs on R:\.
