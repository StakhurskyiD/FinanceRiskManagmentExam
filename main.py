import pandas as pd
from sklearn.cluster import KMeans
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from clustering import final_df


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


# Function to calculate the minimum risk portfolio for each cluster
def calculate_min_risk_portfolio(df, returns_df):
    portfolios = []
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        tickers = cluster_data['Company']

        if len(tickers) == 0:
            continue

        cluster_returns = returns_df[tickers]

        # Calculate the covariance matrix
        cov_matrix = cluster_returns.cov()

        # Number of assets
        n = len(tickers)

        # Define the optimization variables
        weights = cp.Variable(n)

        # Define the objective function (portfolio variance)
        portfolio_variance = cp.quad_form(weights, cov_matrix)

        # Define the constraints
        constraints = [cp.sum(weights) == 1, weights >= 0]

        # Define the optimization problem
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)

        # Solve the problem
        problem.solve()

        # Extract the weights
        min_risk_weights = weights.value

        # Store the portfolio
        portfolio = {'Cluster': cluster, 'Tickers': tickers.tolist(), 'Weights': min_risk_weights.tolist()}
        portfolios.append(portfolio)

    return portfolios


# Function to construct VAWI
def construct_vawi(portfolio, returns_df):
    tickers = portfolio['Tickers']
    weights = np.array(portfolio['Weights'])

    # Calculate the portfolio weekly returns
    portfolio_returns = returns_df[tickers].dot(weights)

    # Initialize VAWI
    vawi = [1000]

    for r in portfolio_returns:
        vawi.append(vawi[-1] * (1 + r / 100))  # Assuming returns are in percentage

    return vawi[1:]  # Exclude the initial value


# Function to calculate K-ratio
def calculate_k_ratio(vawi):
    cumulative_return = vawi[-1] - vawi[0]
    std_cumulative_return = np.std(vawi)
    k_ratio = cumulative_return / std_cumulative_return
    return k_ratio


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

# Filter out companies with ESG Score of 0
filtered_df = merged_df[merged_df['ESG Score'] != 0]

# Prepare data for clustering and drop rows with NaN values
clustering_data = filtered_df[['Risk (Std Dev)', 'Expected Return', 'ESG Score']].dropna()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(clustering_data)

# Merge clustering results back to the filtered DataFrame
filtered_df = filtered_df.merge(clustering_data[['Risk (Std Dev)', 'Expected Return', 'ESG Score', 'Cluster']],
                                on=['Risk (Std Dev)', 'Expected Return', 'ESG Score'],
                                how='left')

# Calculate the minimum risk portfolio for each cluster
portfolios = calculate_min_risk_portfolio(filtered_df, returns_df)

# Construct VAWI and calculate K-ratio for each portfolio
vawi_data = []
for portfolio in portfolios:
    vawi = construct_vawi(portfolio, returns_df)
    k_ratio = calculate_k_ratio(vawi)
    vawi_data.append({
        'Cluster': portfolio['Cluster'],
        'VAWI': vawi,
        'K-ratio': k_ratio
    })

    # Plot VAWI for each portfolio
    plt.figure()
    plt.plot(vawi)
    plt.title(f'Cluster {portfolio["Cluster"]} - VAWI')
    plt.xlabel('Week')
    plt.ylabel('VAWI')
    plt.savefig(f'output/vawi_cluster_{portfolio["Cluster"]}.png')

    # Plot portfolio weights for each cluster
    plt.figure()
    plt.bar(portfolio['Tickers'], portfolio['Weights'])
    plt.title(f'Cluster {portfolio["Cluster"]} - Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weights')
    plt.xticks(rotation=45)
    plt.savefig(f'output/weights_cluster_{portfolio["Cluster"]}.png')

# Save the final DataFrame and portfolios to a new CSV file
output_file_path = 'output/risk_return_esg_clusters_output.csv'
final_df.to_csv(output_file_path, index=False)

# Save the portfolios to a new CSV file
portfolios_df = pd.DataFrame(portfolios)
portfolios_output_file_path = 'output/portfolios_output.csv'
portfolios_df.to_csv(portfolios_output_file_path, index=False)

# Save VAWI and K-ratio data to a new CSV file
vawi_output_file_path = 'output/vawi_k_ratio_output.csv'
vawi_df = pd.DataFrame(vawi_data)
vawi_df.to_csv(vawi_output_file_path, index=False)

print(f"Risk, return, ESG scores, and cluster data have been calculated and saved to '{output_file_path}'")
print(f"Minimum risk portfolios have been calculated and saved to '{portfolios_output_file_path}'")
print(f"VAWI and K-ratio data have been calculated and saved to '{vawi_output_file_path}'")
print("Charts have been saved to the output directory")