"""Check healthy ranges in CSV"""
import pandas as pd
import numpy as np

df = pd.read_csv('models/kidney_disease.csv', na_values=['?', '\t?', ' ', '', '\t'])
df['classification'] = df['classification'].str.strip()
healthy = df[df['classification'] == 'notckd']

print('HEALTHY RANGES (notckd):')
print('=' * 60)
for col in ['age', 'bp', 'sg', 'al', 'bu', 'sc', 'bgr', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
    if col in healthy.columns:
        col_data = pd.to_numeric(healthy[col], errors='coerce')
        col_data = col_data.dropna()
        if len(col_data) > 0:
            print(f'{col}: {col_data.min():.2f} - {col_data.max():.2f} (mean: {col_data.mean():.2f})')

print('\nOUR TEST VALUES:')
print('=' * 60)
print('age: 35')
print('bp: 120')
print('sg: 1.015')
print('al: 0')
print('bu: 30')
print('sc: 0.9')
print('bgr: 100')
print('sod: 140')
print('pot: 4.5')
print('hemo: 15')
print('pcv: 45')
print('wc: 7000')
print('rc: 4.5')

