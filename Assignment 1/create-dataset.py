import pandas as pd
import os

def process_data(input_path, output_filename='dataBank_transformed.csv'):
    """
    Reads data from a CSV file, performs Min-Max normalization on numerical columns,
    performs one-hot encoding on categorical columns, renames columns, and 
    STRICTLY reindexes the dataframe to match a specific target header structure.
    
    It logs any columns that are present in the target header but missing from the
    input data (which will be filled with 0).

    Args:
        input_path (str): Path to the input CSV file (e.g., 'bank-additional-full.csv').
        output_filename (str): The name of the file to save the transformed data to.
    
    Returns:
        str: The comma-separated header string of the transformed columns.
    """
    
    # --- 1. Define the EXACT Target Header List ---
    target_columns = [
        "age", "job=housemaid", "job=services", "job=admin.", "job=blue-collar", 
        "job=technician", "job=retired", "job=management", "job=unemployed", 
        "job=self-employed", "job=unknown", "job=entrepreneur", "job=student", 
        "marital=married", "marital=single", "marital=divorced", "marital=unknown", 
        "education=basic.4y", "education=high.school", "education=basic.6y", 
        "education=basic.9y", "education=professional.course", "education=unknown", 
        "education=university.degree", "education=illiterate", "default=0", 
        "default=unknown", "default=1", "housing=0", "housing=1", "housing=unknown", 
        "loan=0", "loan=1", "loan=unknown", "contact=cellular", "month=may", 
        "month=jun", "month=jul", "month=aug", "month=oct", "month=nov", 
        "month=dec", "month=mar", "month=apr", "month=sep", "day_of_week=mon", 
        "day_of_week=tue", "day_of_week=wed", "day_of_week=thu", "day_of_week=fri", 
        "duration", "campaign", "pdays", "previous", "poutcome=nonexistent", 
        "poutcome=failure", "poutcome=success", "emp.var.rate", "cons.price.idx", 
        "cons.conf.idx", "euribor3m", "nr.employed", "class"
    ]

    # --- 2. Read and Prepare Data ---
    print(f"Reading data from: {input_path}")
    try:
        df = pd.read_csv(input_path, sep=';', quotechar='"')
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found. Please check the file path.")
        return None

    df = df.rename(columns={'y': 'class'})

    # Map 'yes'/'no' in the target class to 1/0
    if 'class' in df.columns:
        df['class'] = df['class'].map({'yes': 1, 'no': 0})

    # --- 3. Normalize Numerical Columns (Min-Max Scaling) ---
    # Based on the sample data, numerical features are scaled to range [0, 1]
    numeric_cols = [
        'age', 'duration', 'campaign', 'pdays', 'previous', 
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
        'euribor3m', 'nr.employed'
    ]
    
    print("Normalizing numerical columns...")
    for col in numeric_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            # Avoid division by zero if max == min
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0

    # --- 4. One-Hot Encoding ---
    categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 
        'loan', 'contact', 'month', 'day_of_week', 'poutcome'
    ]
    # dtype=int ensures that the output values are 0 and 1 instead of False and True
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)

    # --- 5. Rename Columns to Match Target Format ---
    column_mapping = {}
    for col in df_encoded.columns:
        final_col = col
        
        # Convert "prefix_value" to "prefix=value"
        for prefix in categorical_cols:
            if col.startswith(prefix + '_'):
                final_col = col.replace(prefix + '_', prefix + '=', 1)
                break
        
        # specific binary value adjustments
        if 'default=no' in final_col: final_col = final_col.replace('default=no', 'default=0')
        if 'default=yes' in final_col: final_col = final_col.replace('default=yes', 'default=1')
        
        if 'housing=no' in final_col: final_col = final_col.replace('housing=no', 'housing=0')
        if 'housing=yes' in final_col: final_col = final_col.replace('housing=yes', 'housing=1')
        
        if 'loan=no' in final_col: final_col = final_col.replace('loan=no', 'loan=0')
        if 'loan=yes' in final_col: final_col = final_col.replace('loan=yes', 'loan=1')
        
        column_mapping[col] = final_col

    df_renamed = df_encoded.rename(columns=column_mapping)

    # --- 6. LOGGING MISSING COLUMNS ---
    # Check which target columns are NOT in our processed data
    present_columns = set(df_renamed.columns)
    missing_columns = [col for col in target_columns if col not in present_columns]

    if missing_columns:
        print("\n" + "="*50)
        print(f"WARNING: {len(missing_columns)} columns from the target header are MISSING in the data.")
        print("These will be automatically created and filled with 0s:")
        print("-" * 50)
        for col in missing_columns:
            print(f"  [MISSING] {col}")
        print("="*50 + "\n")
    else:
        print("\nSUCCESS: All target columns are present in the input data.\n")

    # --- 7. Strict Reindexing ---
    # Forces EXACT column structure, filling missing ones with 0
    df_final = df_renamed.reindex(columns=target_columns, fill_value=0)

    # --- 8. Save ---
    df_final.to_csv(output_filename, index=False)
    print(f"Successfully saved transformed data to {output_filename}")
    
    return ','.join(df_final.columns)

# --- Main Execution ---
if __name__ == "__main__":
    input_file = "bank-additional-full.csv"
    
    # Check if file exists locally before running
    if os.path.exists(input_file):
        result_header = process_data(input_file, output_filename='dataBank_transformed.csv')
        
        if result_header:
            # Verification
            expected_header = "age,job=housemaid,job=services,job=admin.,job=blue-collar,job=technician,job=retired,job=management,job=unemployed,job=self-employed,job=unknown,job=entrepreneur,job=student,marital=married,marital=single,marital=divorced,marital=unknown,education=basic.4y,education=high.school,education=basic.6y,education=basic.9y,education=professional.course,education=unknown,education=university.degree,education=illiterate,default=0,default=unknown,default=1,housing=0,housing=1,housing=unknown,loan=0,loan=1,loan=unknown,contact=cellular,month=may,month=jun,month=jul,month=aug,month=oct,month=nov,month=dec,month=mar,month=apr,month=sep,day_of_week=mon,day_of_week=tue,day_of_week=wed,day_of_week=thu,day_of_week=fri,duration,campaign,pdays,previous,poutcome=nonexistent,poutcome=failure,poutcome=success,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed,class"
            
            if result_header == expected_header:
                print("Header verification: PASSED")
            else:
                print("Header verification: FAILED (Output does not match target)")
    else:
        print(f"File not found: {input_file}")
        print("Please ensure 'bank-additional-full.csv' is in the current directory.")