import sys
from pathlib import Path
p = Path('backend/src/agriconnect/services/data_collection/documents/sonagess.py')
if not p.exists():
    print('MISSING')
    sys.exit(2)
data = p.read_bytes()
# UTF-8 BOM
bom = b"\xef\xbb\xbf"
if data.startswith(bom):
    p.write_bytes(data[len(bom):])
    print('BOM_REMOVED')
else:
    # Also remove any potential U+FEFF at UTF-16/other encodings if present as first char when decoded
    try:
        s = data.decode('utf-8')
    except Exception:
        # fallback: try to replace Unicode FEFF if present at start after decoding with errors='replace'
        s = data.decode('utf-8', errors='replace')
    if s and s[0] == '\ufeff':
        s2 = s.lstrip('\ufeff')
        p.write_text(s2, encoding='utf-8')
        print('UNICODE_FEFF_REMOVED')
    else:
        print('NO_BOM_FOUND')
