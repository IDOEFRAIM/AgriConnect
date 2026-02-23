from pathlib import Path
p=Path('backend/src/agriconnect/services/data_collection/weather/documents_meteo.py')
b=p.read_bytes()
if b.startswith(b'\xef\xbb\xbf'):
    p.write_bytes(b[len(b'\xef\xbb\xbf'):])
    print('BOM removed')
else:
    print('No BOM')
