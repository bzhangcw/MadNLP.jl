import pandas as pd
import sys

# Load the single CSV file
file_path = sys.argv[1]


# Read the CSV file
df = pd.read_csv(file_path)

# Identify the unique methods
methods = df['method'].unique()

# Create a dictionary to hold dataframes for each method
dfs = {}

# Split the dataframe into separate dataframes for each method
for method in methods:
    dfs[method] = df[df['method'] == method].copy()
    # Rename columns to include the method information
    dfs[method].columns = [f"{col}_{method}" if col not in ['name', 'method'] else col for col in dfs[method].columns]

# Merge the dataframes on 'name'
merged_df = dfs[methods[0]].drop(columns=['method'])
for method in methods[1:]:
    merged_df = pd.merge(merged_df, dfs[method].drop(columns=['method']), on='name', how='outer')

# Define a function to compute bool_success, k, and t
def compute_success_and_metrics(row, method):
    primal_feas = row[f'primal_feas_{method}']
    dual_feas = row[f'dual_feas_{method}']
    iter = row[f'iter_{method}']
    total_time = row.get(f'total_time_{method}', 10000)  # Handle possible missing 'total_time' column
    
    bool_success = (primal_feas < 1e-6) & (dual_feas < 1e-6)
    k = iter if bool_success else 10000
    t = total_time if bool_success else 10000
    
    return bool_success, k, t

# Apply the function to compute the new metrics for each method
for method in methods:
    merged_df[f'bool_success_{method}'], merged_df[f'k_{method}'], merged_df[f't_{method}'] = zip(*merged_df.apply(compute_success_and_metrics, method=method, axis=1))

# Select the desired columns
output_columns = ['nvar', 'ncon', 'nnzj', 'neq', 'primal_feas', 'dual_feas', 'bool_success', 'k', 't', 'objective']
output_df = merged_df[['name'] + [f'{col}_{method}' for method in methods for col in output_columns]]

# Create a MultiIndex for the columns to represent nested columns
output_df.columns = pd.MultiIndex.from_tuples(
    [('name', '')] + [(col.split('_')[-1], '_'.join(col.split('_')[:-1])) for col in output_df.columns if col != 'name'],
    names=['metric', 'method']
)

# Save the reshaped dataframe to an Excel file with multirow and multicol
excel_file_path = 'reshaped_metrics.xlsx'
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    output_df.to_excel(writer, sheet_name='Metrics', merge_cells=True)

print(f"Dataframe saved to {excel_file_path}")