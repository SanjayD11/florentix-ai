import requests

url = 'https://openrouter.ai/api/v1/models'
try:
    models = requests.get(url).json()['data']
    free_vision = [m['id'] for m in models if m['pricing']['prompt'] == '0' and m.get('architecture', {}).get('modality') and 'image' in m['architecture']['modality']]
    print('Free Vision Models:', free_vision)
except Exception as e:
    print(f'Error: {e}')
