import base64
import os

img_path = r'C:\Users\lenovo\.gemini\antigravity\brain\bc687331-9a4a-4622-b93d-57e8a7f3fb21\florentix_favicon_geometric_1775986000477.png'

with open(img_path, 'rb') as f:
    img_data = f.read()

b64 = base64.b64encode(img_data).decode('utf-8')

svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <image href="data:image/png;base64,{b64}" width="1024" height="1024"/>
</svg>'''

out_path_1 = r'd:\Antigravity Projects\cloned plant disease prediction ai\plant-disease-prediction-ai\frontend\favicon.svg'
out_path_2 = r'd:\Antigravity Projects\cloned plant disease prediction ai\plant-disease-prediction-ai\frontend\logo.svg'

with open(out_path_1, 'w') as f:
    f.write(svg_content)
with open(out_path_2, 'w') as f:
    f.write(svg_content)

print(f"Successfully generated SVGs.")
