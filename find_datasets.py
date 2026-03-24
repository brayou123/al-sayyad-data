#!/usr/bin/env python3
import os
from copernicusmarine import describe

USERNAME = os.environ.get("COPERNICUS_USER")
PASSWORD = os.environ.get("COPERNICUS_PASS")

print("🔍 Searching for Mediterranean datasets...")

# البحث عن مجموعات فيزياء البحر المتوسط
result = describe(
    contains="MEDSEA",
    username=USERNAME,
    password=PASSWORD
)

print("\n📊 Found datasets:")
for product in result.products:
    if hasattr(product, 'datasets'):
        for dataset in product.datasets:
            print(f"  - {dataset.dataset_id}")
            # إظهار المتغيرات المتاحة أيضاً
            if hasattr(dataset, 'variables'):
                print(f"    Variables: {dataset.variables}")
