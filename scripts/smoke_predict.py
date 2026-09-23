import io
import sys
import httpx
from PIL import Image

img = Image.new('RGB', (224, 224), color=(120, 120, 120))
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)
files = {'file': ('test.jpg', buf.getvalue(), 'image/jpeg')}
try:
    res = httpx.post('http://127.0.0.1:8000/predict', files=files, timeout=10.0)
    print(res.status_code)
    print(res.json())
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
