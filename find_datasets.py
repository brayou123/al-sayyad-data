#!/usr/bin/env python3
import os
from copernicusmarine import describe

print("🔍 Searching for Mediterranean datasets...")

# المتغيرات البيئية ستُقرأ تلقائياً
result = describe(contains="MED")

print("\n📊 Found datasets:")
for product in result.products:
    if hasattr(product, 'datasets'):
        for dataset in product.datasets:
            print(f"  - {dataset.dataset_id}")
            if hasattr(dataset, 'variables'):
                print(f"    Variables: {dataset.variables}")
