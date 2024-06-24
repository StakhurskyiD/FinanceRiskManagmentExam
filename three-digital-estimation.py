import pandas as pd


# Function to calculate risk (standard deviation of returns) and expected return
def calculate_risk_return(df):
    risk_return_data = []
    company_abbreviations = df.columns[1:]  # Exclude the 'Date' column

    for company in company_abbreviations:
        # Remove '%' symbol and convert to float
        df[company] = df[company].str.replace('%', '').astype(float)

        # Calculate risk (standard deviation of returns) and expected return
        risk = df[company].std()
        expected_return = df[company].mean()
        risk_return_data.append({'Company': company, 'Risk (Std Dev)': risk, 'Expected Return': expected_return})

    return pd.DataFrame(risk_return_data)


# Function to merge risk, return, and ESG scores
def merge_risk_return_esg(risk_return_df, esg_df):
    esg_df.columns = ['Company', 'ESG Score']  # Ensure ESG columns are named correctly
    merged_df = pd.merge(risk_return_df, esg_df, on='Company', how='left')
    return merged_df


# Function to form a 3-digital estimation
def form_three_digital_estimation(df):
    df['3-Digit Estimation'] = df.apply(
        lambda row: f"{row['ESG Score']:.1f},{row['Expected Return']:.2f},{row['Risk (Std Dev)']:.2f}", axis=1)
    return df


# Load the CSV files
returns_file_path = 'weekly-returns.csv'  # Update with your returns file path
esg_file_path = 'esg_scores.csv'  # Update with your ESG scores file path

# Read the CSV file with weekly returns, skipping the first row which contains company names
returns_df = pd.read_csv(returns_file_path, skiprows=1)

# Read the CSV file with ESG scores
esg_df = pd.read_csv(esg_file_path)

# Calculate risk and return
risk_return_df = calculate_risk_return(returns_df)

# Merge risk, return, and ESG scores
merged_df = merge_risk_return_esg(risk_return_df, esg_df)

# Form the 3-digital estimation
final_df = form_three_digital_estimation(merged_df)

# Save the final DataFrame to a new CSV file
output_file_path = 'output/risk_return_esg_output.csv'
final_df.to_csv(output_file_path, index=False)

print(f"Risk, return, and ESG scores data have been calculated and saved to '{output_file_path}'")