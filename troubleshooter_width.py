import os
import pandas as pd

folder = 'spacetimeformer/data/oos'
width_95_files = []
width_96_files = []

# Group files based on their tensor width
for filename in os.listdir(folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder, filename)
        dataset = pd.read_csv(filepath)
        width = dataset.shape[1]  # Number of columns
        if width == 95:
            width_95_files.append(filename)
        elif width == 96:
            width_96_files.append(filename)

# Load the first file from each group to compare column names
if width_95_files and width_96_files:
    file_95 = pd.read_csv(os.path.join(folder, width_95_files[0]))
    file_96 = pd.read_csv(os.path.join(folder, width_96_files[0]))

    columns_95 = set(file_95.columns)
    columns_96 = set(file_96.columns)

    # Find columns present in 96 but not in 95
    extra_columns = columns_96 - columns_95
    print(f"Columns in 96-width files but not in 95-width files: {extra_columns}")

else:
    print("Insufficient data to compare.")
