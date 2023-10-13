import pandas as pd

# Read the filter values from another Excel file into a pandas Series
filter_values_series = pd.read_excel('Calificación Pipeline.xlsx', engine='openpyxl', sheet_name='Hoja 1', header=1, squeeze=True)
print(filter_values_series.columns)

# Convert the filter values Series to a list
filter_values_list = filter_values_series['ID_conversation'].tolist()

# Read the main Excel file into a pandas DataFrame
df = pd.read_excel('Base_Consoldada_Febrero_Bad_C_SAT_Snooze_Ambas.xlsx', engine='openpyxl')

# Filter the DataFrame by a specific column and a list of values
filtered_df = df[df['ID'].isin(filter_values_list)]

# Save the filtered data to a new Excel file
filtered_df.to_excel('Base_Consoldada_Febrero_Bad_C_SAT_Snooze_Ambas_filtered.xlsx', index=False)

# Print a message to confirm that the output file was created
print('Filtered data saved to output_filename.xlsx')
