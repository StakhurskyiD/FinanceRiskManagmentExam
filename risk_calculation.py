import pandas as pd

# Load the CSV file
file_path = 'weekly-returns.csv'  # Update this with the actual file path

# Read the CSV file into a DataFrame, skipping the first row which contains company names
df = pd.read_csv(file_path, skiprows=1)

# Get the company abbreviations from the first row (headers)
company_abbreviations = df.columns[1:]  # Exclude the 'Date' column

# Initialize a list to store risk and return data
risk_return_data = []

# Process each company's data
for company in company_abbreviations:
    # Remove '%' symbol and convert to float
    df[company] = df[company].str.replace('%', '').astype(float)

    # Calculate risk (standard deviation of returns) and expected return for each company
    risk = df[company].std()
    expected_return = df[company].mean()
    risk_return_data.append({'Company': company, 'Risk (Std Dev)': risk, 'Expected Return': expected_return})

# Save the risk and expected return data to a new CSV file
risk_return_df = pd.DataFrame(risk_return_data)
risk_return_df.to_csv('risk_return_output.csv', index=False)

print("Risk and expected return data have been calculated and saved to 'risk_return_output.csv'")