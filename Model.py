import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import os

# 1. Data Loading and Initial Processing
file_path = #Insert Path

df = pd.read_excel(file_path, sheet_name="Data", usecols="A:W")

# print("Original column names in df:")
# print(df.columns.tolist())

# 2. Define scenario parameters
scenarios = {
    'Scenario_4.1': {
        'SAV': {'cost': 240, 'travel_time': 40, 'waiting_time': 10, 'total_time': 50},
        'Taxi/solo-hailing': {'cost': 300, 'travel_time': 35, 'waiting_time': 5, 'total_time': 40},
        'Metro': {'cost': 110, 'travel_time': 50, 'waiting_time': 15, 'total_time': 65},
        'Bus': {'cost': 50, 'travel_time': 75, 'waiting_time': 15, 'total_time': 90}
    },
    'Scenario_4.2': {
        'SAV': {'cost': 50, 'travel_time': 35, 'waiting_time': 3, 'total_time': 38},
        'Taxi/solo-hailing': {'cost': 90, 'travel_time': 25, 'waiting_time': 5, 'total_time': 30},
        'Metro': {'cost': 15, 'travel_time': 35, 'waiting_time': 10, 'total_time': 45},
        'Bus': {'cost': 12, 'travel_time': 40, 'waiting_time': 15, 'total_time': 55}
    },
    'Scenario_4.3': {
        'SAV': {'cost': 20, 'travel_time': 7, 'waiting_time': 3, 'total_time': 10},
        'Taxi/solo-hailing': {'cost': 40, 'travel_time': 5, 'waiting_time': 2, 'total_time': 7},
        'Metro': {'cost': 5, 'travel_time': 10, 'waiting_time': 10, 'total_time': 20},
        'Bus': {'cost': 7, 'travel_time': 15, 'waiting_time': 5, 'total_time': 20}
    },
    'Scenario_4.4': {
        'SAV': {'cost': 40, 'travel_time': 16, 'waiting_time': 5, 'total_time': 21},
        'Taxi/solo-hailing': {'cost': 50, 'travel_time': 10, 'waiting_time': 4, 'total_time': 14},
        'Metro': {'cost': 10, 'travel_time': 10, 'waiting_time': 10, 'total_time': 20},
        'Bus': {'cost': 5, 'travel_time': 15, 'waiting_time': 15, 'total_time': 30}
    }
}


# 3. Encoding mappings
ENCODING_MAPPINGS = {
    'Gender': {'Male': 0, 'Female': 1},
    'Age': {'Under 18': 9, '18-24': 21, '25-34': 29.5, '35-44': 39.5, '45-54': 49.5, '55-64': 59.5, '65 and above': 65},
    'Income': {
        'Less than $10400': 5200,
        '$10400–$14300': 12350,
        '$14300–$19800': 17050,
        '$19800–$31100': 25450,
        '$31100–$48500': 39800,
        '$48500 or more': 48500,
        'Unemployed / Prefer not to say': 0
    },
    'Education': {
        'Junior High School Graduate': 0,
        'High School / Vocational High School Graduate': 1,
        'Associate Degree / Vocational College Graduate': 2,
        'Bachelor\'s Degree': 3,
        'Master\'s Degree': 4,
        'Doctorate and above': 5
    },
    'Household_Size': {
        '1 person': 1,
        '2 people': 2,
        '3 people': 3,
        '4 or more people': 4
    },
    'Car_Ownership': {'No': 0, 'Yes': 1},
    'PT_Accessibility': {
        'Neither': 0,
        'Good accessibility to the bus': 1,
        'Good accessibility to the metro': 2,
        'Both are good': 3
    },
    'Shared_Mobility_Frequency': {
        'Never': 0,
        'Occasionally': 1,
        '1-2 times per week': 2,
        '3 or more times per week': 3
    },
    'AV_Experience': {'No': 0, 'Yes': 1},
    'AV_Willingness': {'Unwilling': 0, 'Willing': 1},
    'SM_Experience': {'No': 0, 'Yes': 1},
    'SM_Willingness': {'Unwilling': 0, 'Willing': 1}
}

# 4. Process scenario responses with all variables
def process_scenario_responses(df, scenarios):
    processed_data = []
    
    # print(f"Initial df shape: {df.shape}")
    
    for idx, row in df.iterrows():
        for scenario_name in ['Scenario_4.1', 'Scenario_4.2', 'Scenario_4.3', 'Scenario_4.4']:
            chosen_mode = row[scenario_name]
            scenario_params = scenarios[scenario_name]
            
            for mode in ['SAV', 'Taxi/solo-hailing', 'Metro', 'Bus']:
                data_point = {
                    'respondent_id': idx,
                    'scenario': scenario_name,
                    'mode': mode,
                    'cost': scenario_params[mode]['cost'],
                    'travel_time': scenario_params[mode]['travel_time'],
                    'waiting_time': scenario_params[mode]['waiting_time'],
                    'total_time': scenario_params[mode]['total_time'],
                    'chosen': 1 if mode == chosen_mode else 0
                }
                
                # Add all demographic and behavioral variables
                for col in df.columns:
                    if col not in ['Scenario_4.1', 'Scenario_4.2', 'Scenario_4.3', 'Scenario_4.4']:
                        data_point[col] = row[col]
                
                processed_data.append(data_point)
    
    result_df = pd.DataFrame(processed_data)
    # print(f"Processed data shape: {result_df.shape}")
    # print("Columns in processed data:", result_df.columns.tolist())
    return result_df

# 5. Process all variables
def encode_all_variables(df):
    encoded_df = df.copy()
    inverse_mappings = {}   # {encoded_col_name: {encoded_value: original_label}}

    for col, mapping in ENCODING_MAPPINGS.items():
        if col in encoded_df.columns:
            unique_values = encoded_df[col].unique()
            if len(unique_values) == 1:
                encoded_df[f'{col}_encoded'] = 1
                inverse_mappings[f'{col}_encoded'] = {1: str(unique_values[0])}
            else:
                encoded_df[f'{col}_encoded'] = encoded_df[col].map(mapping)
                inverse_mappings[f'{col}_encoded'] = {v: k for k, v in mapping.items()}

    categorical_cols = ['Major', 'Region']
    for col in categorical_cols:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[f'{col}_encoded'] = le.fit_transform(encoded_df[col])
            inverse_mappings[f'{col}_encoded'] = dict(enumerate(le.classes_))

    return encoded_df, inverse_mappings

def prepare_data_for_analysis():
    scenario_data = process_scenario_responses(df, scenarios)
    encoded_data, inverse_mappings = encode_all_variables(scenario_data)

    basic_features = ['cost', 'travel_time', 'waiting_time']

    individual_vars = [
        'Gender_encoded', 'Age_encoded', 'Income_encoded', 'Education_encoded',
        'Household_Size_encoded', 'Car_Ownership_encoded',
        'PT_Accessibility_encoded', 'Shared_Mobility_Frequency_encoded',
        'AV_Experience_encoded', 'SM_Experience_encoded', 'SM_Willingness_encoded',
        'Major_encoded', 'Region_encoded'
    ]
    individual_vars = [v for v in individual_vars if v in encoded_data.columns]

    # Drop rows with NA in any of these
    encoded_data = encoded_data.dropna(subset=basic_features + individual_vars).copy()

    # Alternative-specific constants (Bus = reference)
    encoded_data['ASC_SAV']   = (encoded_data['mode'] == 'SAV').astype(int)
    encoded_data['ASC_Taxi']  = (encoded_data['mode'] == 'Taxi/solo-hailing').astype(int)
    encoded_data['ASC_Metro'] = (encoded_data['mode'] == 'Metro').astype(int)

    # Demographic interactions with each non-reference alternative
    interaction_cols = []
    for v in individual_vars:
        for alt in ['SAV', 'Taxi', 'Metro']:
            col = f'{v}_x_{alt}'
            encoded_data[col] = encoded_data[v] * encoded_data[f'ASC_{alt}']
            interaction_cols.append(col)

    feature_columns = (
        ['ASC_SAV', 'ASC_Taxi', 'ASC_Metro']
        + basic_features
        + interaction_cols
    )

    # Group identifier: one choice situation = one (respondent, scenario)
    encoded_data['group_id'] = (
        encoded_data['respondent_id'].astype(str) + '_' + encoded_data['scenario']
    )

    X = encoded_data[feature_columns].values.astype(float)
    y = encoded_data['chosen'].values.astype(float)
    group_ids = encoded_data['group_id'].values

    # Scale only the LOS variables (cost, tt, wt). ASCs and interactions stay as-is.
    scaler = StandardScaler()
    los_idx = [feature_columns.index(c) for c in basic_features]
    X[:, los_idx] = scaler.fit_transform(X[:, los_idx])

    return X, y, encoded_data, scaler, feature_columns, group_ids, inverse_mappings

def train_and_evaluate_model(X, y, feature_names, group_ids):
    # Stratified split by group to keep all 4 alternatives of each situation together
    unique_groups = np.unique(group_ids)
    train_g, test_g = train_test_split(unique_groups, test_size=0.2, random_state=42)
    train_mask = np.isin(group_ids, train_g)
    test_mask  = np.isin(group_ids, test_g)

    result = optimize.minimize(
        fun=log_likelihood,
        x0=np.zeros(X.shape[1]),
        args=(X[train_mask], y[train_mask], group_ids[train_mask]),
        method='BFGS',
        options={'maxiter': 1000}
    )

    print("\nModel Coefficients:")
    for name, w in zip(feature_names, result.x):
        print(f"{name:40s} {w:+.4f}")
    print(f"\nLog Likelihood: {-result.fun:.4f}")
    print(f"Converged: {result.success}, iters: {result.nit}")

    return result.x, X[test_mask], y[test_mask]

def log_likelihood(weights, X, y, group_ids):
    u = X @ weights
    df = pd.DataFrame({'u': u, 'y': y, 'g': group_ids})
    df['u'] = df['u'] - df.groupby('g')['u'].transform('max')
    df['eu'] = np.exp(df['u'])
    df['p'] = df['eu'] / df.groupby('g')['eu'].transform('sum')
    return -np.sum(df['y'] * np.log(df['p'] + 1e-12))

def _decode(v, mapping):
    if not mapping:
        return f'{v:g}'
    # Try direct, float, and int lookups
    for key in (v, float(v), int(v) if float(v).is_integer() else None):
        if key is not None and key in mapping:
            return str(mapping[key])
    return f'{v:g}'


# ============================================================
# MODEL VALIDATION FUNCTIONS
# ============================================================

def calculate_standard_errors(weights, X, y, group_ids):
    """
    Calculate standard errors using the Hessian matrix (inverse of Fisher information).
    Returns standard errors, t-values, and p-values for each coefficient.
    """
    from scipy.optimize import approx_fprime
    
    # Compute the Hessian (Fisher Information Matrix) numerically
    def neg_log_likelihood_grad(w):
        """Gradient of negative log-likelihood"""
        u = X @ w
        df = pd.DataFrame({'u': u, 'y': y, 'g': group_ids})
        df['u'] = df['u'] - df.groupby('g')['u'].transform('max')
        df['eu'] = np.exp(df['u'])
        df['p'] = df['eu'] / df.groupby('g')['eu'].transform('sum')
        
        # Gradient calculation
        grad = np.zeros(len(w))
        for i, w_i in enumerate(w):
            df[f'diff_{i}'] = df['p'] * (X[:, i] - df.groupby('g').apply(
                lambda g: (g['p'] * g[X.columns.tolist() if hasattr(X, 'columns') else 
                      list(range(X.shape[1]))].iloc[:, i]).sum()
            ).reindex(df['g']).values)
            grad[i] = -df['y'] * (df[f'diff_{i}'] - df.groupby('g').apply(
                lambda g: (g['p'] * g[X.columns.tolist() if hasattr(X, 'columns') else 
                      list(range(X.shape[1]))].iloc[:, i]).sum()
            ).reindex(df['g']).values).sum()
        
        return grad
    
    # Simplified Hessian approximation using finite differences
    n_params = len(weights)
    hessian = np.zeros((n_params, n_params))
    epsilon = 1e-5
    
    # Compute gradient at the optimum
    def ll(w):
        u = X @ w
        df = pd.DataFrame({'u': u, 'y': y, 'g': group_ids})
        df['u'] = df['u'] - df.groupby('g')['u'].transform('max')
        df['eu'] = np.exp(df['u'])
        df['p'] = df['eu'] / df.groupby('g')['eu'].transform('sum')
        return np.sum(df['y'] * np.log(df['p'] + 1e-12))
    
    # Numerical Hessian
    for i in range(n_params):
        for j in range(n_params):
            w1 = weights.copy()
            w2 = weights.copy()
            w1[i] += epsilon
            w1[j] += epsilon
            w2[i] += epsilon
            w2[j] -= epsilon
            w3 = weights.copy()
            w3[i] -= epsilon
            w3[j] += epsilon
            w4 = weights.copy()
            w4[i] -= epsilon
            w4[j] -= epsilon
            
            hessian[i, j] = (ll(w1) - ll(w2) - ll(w3) + ll(w4)) / (4 * epsilon * epsilon)
    
    # Try to invert the Hessian
    try:
        cov_matrix = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        # If Hessian is singular, use pseudo-inverse
        cov_matrix = np.linalg.pinv(hessian)
        std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
    
    # Calculate t-values and p-values
    t_values = weights / (std_errors + 1e-10)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=len(y) - n_params))
    
    return std_errors, t_values, p_values


def calculate_goodness_of_fit(weights, X, y, group_ids):
    """
    Calculate goodness of fit metrics:
    - McFadden's Pseudo R²
    - Hit Ratio (prediction accuracy)
    - Log-Likelihood
    """
    # Log-likelihood of the model
    u = X @ weights
    df = pd.DataFrame({'u': u, 'y': y, 'g': group_ids})
    df['u'] = df['u'] - df.groupby('g')['u'].transform('max')
    df['eu'] = np.exp(df['u'])
    df['p'] = df['eu'] / df.groupby('g')['eu'].transform('sum')
    ll_model = np.sum(df['y'] * np.log(df['p'] + 1e-12))
    
    # Log-likelihood of null model (only ASCs)
    n_params_null = 3  # ASC_SAV, ASC_Taxi, ASC_Metro
    weights_null = np.zeros(X.shape[1])
    # Set ASCs to equal shares
    for i, col in enumerate(feature_columns):
        if 'ASC' in col:
            weights_null[i] = np.log(0.25)  # Equal probability
    
    u_null = X @ weights_null
    df_null = pd.DataFrame({'u': u_null, 'y': y, 'g': group_ids})
    df_null['u'] = df_null['u'] - df_null.groupby('g')['u'].transform('max')
    df_null['eu'] = np.exp(df_null['u'])
    df_null['p'] = df_null['eu'] / df_null.groupby('g')['eu'].transform('sum')
    ll_null = np.sum(df_null['y'] * np.log(df_null['p'] + 1e-12))
    
    # McFadden's Pseudo R²
    mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0
    
    # Hit Ratio - predict choices
    df['predicted'] = df.groupby('g')['p'].transform(lambda x: x == x.max()).astype(int)
    hit_ratio = (df['y'] * df['predicted']).sum() / df['y'].sum()
    
    return {
        'log_likelihood_model': ll_model,
        'log_likelihood_null': ll_null,
        'mcfadden_r2': mcfadden_r2,
        'hit_ratio': hit_ratio,
        'n_observations': len(y),
        'n_parameters': len(weights)
    }


def create_validation_visualizations(weights, feature_columns, X, y, group_ids, 
                                     std_errors, t_values, p_values, goodness_of_fit):
    """
    Create model validation visualizations:
    1. Coefficient significance plot (with t-values and confidence intervals)
    2. Goodness of fit summary
    3. Prediction accuracy plot
    """
    plt.style.use('default')
    
    # 1. Coefficient Significance Plot with t-values
    print("\nGenerating coefficient significance plot...")
    fig, ax = plt.subplots(figsize=(12, max(6, len(feature_columns) * 0.35)))
    
    # Sort by absolute t-value
    sorted_idx = np.argsort(np.abs(t_values))[::-1]
    sorted_names = [feature_columns[i] for i in sorted_idx]
    sorted_weights = weights[sorted_idx]
    sorted_t = t_values[sorted_idx]
    sorted_se = std_errors[sorted_idx]
    sorted_p = p_values[sorted_idx]
    
    y_pos = np.arange(len(sorted_names))
    
    # Plot coefficients with error bars
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in sorted_p]
    ax.barh(y_pos, sorted_weights, xerr=1.96*sorted_se, color=colors, 
            alpha=0.8, capsize=3, error_kw={'elinewidth': 1, 'alpha': 0.5})
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add significance markers
    for i, (w, t, p) in enumerate(zip(sorted_weights, sorted_t, sorted_p)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.annotate(f'{t:+.2f}{sig}', xy=(w, i), xytext=(5, 0), 
                   textcoords='offset points', va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Coefficient Value (with 95% CI)')
    ax.set_title('Coefficient Significance Plot\n(* p<0.05, ** p<0.01, *** p<0.001)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('coefficient_significance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Goodness of Fit Summary
    print("Generating goodness of fit summary...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    MODEL GOODNESS OF FIT                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  McFadden's Pseudo R²:        {goodness_of_fit['mcfadden_r2']:>10.4f}                     ║
    ║  Hit Ratio (Accuracy):        {goodness_of_fit['hit_ratio']:>10.1%}                     ║
    ║  Log-Likelihood (Model):      {goodness_of_fit['log_likelihood_model']:>10.4f}                     ║
    ║  Log-Likelihood (Null):       {goodness_of_fit['log_likelihood_null']:>10.4f}                     ║
    ║  Number of Observations:      {goodness_of_fit['n_observations']:>10d}                     ║
    ║  Number of Parameters:        {goodness_of_fit['n_parameters']:>10d}                     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Interpretation:
    • McFadden's R² > 0.2 indicates good model fit
    • Hit Ratio shows percentage of correctly predicted choices
    """
    
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig('goodness_of_fit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Prediction Probability Distribution
    print("Generating prediction accuracy plot...")
    u = X @ weights
    df = pd.DataFrame({'u': u, 'y': y, 'g': group_ids})
    df['u'] = df['u'] - df.groupby('g')['u'].transform('max')
    df['eu'] = np.exp(df['u'])
    df['p'] = df['eu'] / df.groupby('g')['eu'].transform('sum')
    
    # Separate chosen vs not chosen probabilities
    chosen_probs = df[df['y'] == 1]['p']
    not_chosen_probs = df[df['y'] == 0]['p']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(chosen_probs, bins=30, alpha=0.7, label='Chosen Alternative', color='green')
    axes[0].hist(not_chosen_probs, bins=30, alpha=0.5, label='Not Chosen', color='red')
    axes[0].axvline(x=0.25, color='black', linestyle='--', label='Random (25%)')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Predicted Probabilities')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [chosen_probs, not_chosen_probs]
    bp = axes[1].boxplot(data_to_plot, labels=['Chosen', 'Not Chosen'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Predicted Probability')
    axes[1].set_title('Probability Comparison: Chosen vs Not Chosen')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nValidation visualizations saved:")
    print("1. coefficient_significance.png")
    print("2. goodness_of_fit.png")
    print("3. prediction_accuracy.png")


def print_validation_summary(weights, feature_columns, std_errors, t_values, p_values, goodness_of_fit):
    """
    Print a formatted validation summary table.
    """
    print("\n" + "="*80)
    print("MODEL VALIDATION SUMMARY")
    print("="*80)
    
    print("\n{:<35} {:>10} {:>10} {:>10} {:>10}".format(
        "Variable", "Coef", "Std Err", "t-value", "p-value"))
    print("-"*80)
    
    for i, name in enumerate(feature_columns):
        sig = '***' if p_values[i] < 0.001 else '**' if p_values[i] < 0.01 else '*' if p_values[i] < 0.05 else ''
        print("{:<35} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {}".format(
            name, weights[i], std_errors[i], t_values[i], p_values[i], sig))
    
    print("-"*80)
    print("\nGoodness of Fit Metrics:")
    print(f"  McFadden's Pseudo R²:  {goodness_of_fit['mcfadden_r2']:.4f}")
    print(f"  Hit Ratio:             {goodness_of_fit['hit_ratio']:.1%}")
    print(f"  Log-Likelihood:        {goodness_of_fit['log_likelihood_model']:.4f}")
    print(f"  Observations:          {goodness_of_fit['n_observations']}")
    print(f"  Parameters:            {goodness_of_fit['n_parameters']}")
    print("="*80 + "\n")


#Visualization
def create_visualizations(encoded_data, weights, feature_columns, scaler, X_scaled, scenarios):
    plt.style.use('default')
    
    # 1. Coefficient Plot
    print("\nGenerating coefficient plot...")
    print("All coefficients:")
    for name, weight in zip(feature_columns, weights):
        print(f"{name}: {weight:.4f}")

    threshold = 0.0001
    significant_features = [(name, weight) for name, weight in zip(feature_columns, weights) 
                          if abs(weight) > threshold]
    
    if not significant_features:
        print(f"No features found with absolute coefficient value > {threshold}")
        print("Using all features instead.")
        significant_features = list(zip(feature_columns, weights))
    
    names, values = zip(*significant_features)
    
    plt.figure(figsize=(12, max(6, len(names)*0.3)))
    bars = plt.barh(names, values)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        
    plt.title('Feature Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('coefficient_plot.png')
    plt.close()

    # 2. Mode Choice Distribution
    print("Generating mode choice distribution plot...")
    plt.figure(figsize=(10, 6))
    mode_counts = encoded_data.groupby(['scenario', 'mode'])['chosen'].sum()
    mode_counts.unstack().plot(kind='bar', stacked=False)
    plt.title('Mode Choice Distribution Across Scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Number of Choices')
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('mode_choice_distribution.png')
    plt.close()

    # 3. Demographics Impact - Split into separate plots
    print("Generating demographics impact plots...")
    chosen_data = encoded_data[encoded_data['chosen'] == 1]
    
    # Age plot
    plt.figure(figsize=(10, 6))
    age_means = chosen_data.groupby('mode')['Age_encoded'].mean()
    age_means.plot(kind='bar')
    plt.title('Average Age by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.savefig('demographics_age.png')
    plt.close()
    
    # Income plot
    plt.figure(figsize=(10, 6))
    income_means = chosen_data.groupby('mode')['Income_encoded'].mean()
    income_means.plot(kind='bar')
    plt.title('Average Income by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Income')
    plt.tight_layout()
    plt.savefig('demographics_income.png')
    plt.close()
    
    # Education plot
    plt.figure(figsize=(10, 6))
    edu_means = chosen_data.groupby('mode')['Education_encoded'].mean()
    edu_means.plot(kind='bar')
    plt.title('Average Education Level by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Education Level')
    plt.tight_layout()
    plt.savefig('demographics_education.png')
    plt.close()
    
    # Gender plot
    plt.figure(figsize=(10, 6))
    gender_mode = pd.crosstab(chosen_data['mode'], chosen_data['Gender_encoded'])
    gender_mode.plot(kind='bar')
    plt.title('Gender Distribution by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Count')
    plt.legend(['Male', 'Female'])
    plt.tight_layout()
    plt.savefig('demographics_gender.png')
    plt.close()

    # 4. Cost vs Time Trade-off
    print("Generating cost-time trade-off plot...")
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(encoded_data['cost'], 
                        encoded_data['total_time'],
                        c=encoded_data['chosen'],
                        cmap='viridis',
                        alpha=0.6)
    plt.colorbar(scatter, label='Chosen (1) / Not Chosen (0)')
    plt.title('Cost vs Total Time Trade-off')
    plt.xlabel('Cost')
    plt.ylabel('Total Time (minutes)')
    plt.tight_layout()
    plt.savefig('cost_time_tradeoff.png')
    plt.close()

    # 5. Transport Experience Impact - Split into separate plots
    print("Generating transport experience plots...")
    # Shared Mobility Experience
    plt.figure(figsize=(10, 6))
    sm_exp = pd.crosstab(encoded_data['mode'][encoded_data['chosen']==1], 
                        encoded_data['SM_Experience'])
    sm_exp.plot(kind='bar')
    plt.title('Mode Choice by Shared Mobility Experience')
    plt.xlabel('Transport Mode')
    plt.ylabel('Count')
    plt.legend(title='SM Experience')
    plt.tight_layout()
    plt.savefig('shared_mobility_experience.png')
    plt.close()
    
    # AV Experience
    plt.figure(figsize=(10, 6))
    av_exp = pd.crosstab(encoded_data['mode'][encoded_data['chosen']==1], 
                        encoded_data['AV_Experience'])
    av_exp.plot(kind='bar')
    plt.title('Mode Choice by AV Experience')
    plt.xlabel('Transport Mode')
    plt.ylabel('Count')
    plt.legend(title='AV Experience')
    plt.tight_layout()
    plt.savefig('av_experience.png')
    plt.close()

    # 6. Price Sensitivity Analysis - Fixed
    print("Generating price sensitivity plot...")
    plt.figure(figsize=(10, 6))
    price_points = np.linspace(0, 500, 50)
    
    # Get average values for all variables (including constant)
    avg_demographics = X_scaled.mean(axis=0)
    
    sav_probabilities = []
    taxi_probabilities = []
    metro_probabilities = []
    bus_probabilities = []
    
    # Get average cost for each mode from scenarios
    mode_costs = {}
    mode_travel_times = {}
    mode_waiting_times = {}
    for mode in ['SAV', 'Taxi/solo-hailing', 'Metro', 'Bus']:
        costs = []
        travel_times = []
        waiting_times = []
        for scenario_name, scenario_params in scenarios.items():
            costs.append(scenario_params[mode]['cost'])
            travel_times.append(scenario_params[mode]['travel_time'])
            waiting_times.append(scenario_params[mode]['waiting_time'])
        mode_costs[mode] = np.mean(costs)
        mode_travel_times[mode] = np.mean(travel_times)
        mode_waiting_times[mode] = np.mean(waiting_times)
    
    for price in price_points:
        # Create 4 profiles (one for each mode) with average individual characteristics
        # but varying SAV cost while keeping other modes at their average cost
        utilities = []
        for mode_idx, mode_name in enumerate(['SAV', 'Taxi/solo-hailing', 'Metro', 'Bus']):
            # Start with average individual characteristics
            profile = avg_demographics.copy()
            
            # Set cost: vary SAV cost, keep others at average
            if mode_name == 'SAV':
                mode_cost = price  # Vary SAV cost
            else:
                mode_cost = mode_costs[mode_name]  # Keep other modes at average
            
            profile[1] = (mode_cost - scaler.mean_[0]) / scaler.scale_[0]   # cost
            profile[2] = (mode_travel_times[mode_name] - scaler.mean_[1]) / scaler.scale_[1]
            profile[3] = (mode_waiting_times[mode_name] - scaler.mean_[2]) / scaler.scale_[2]       
            
            # Calculate utility for this mode
            utility = np.dot(profile, weights)
            utilities.append(utility)
        
        # Apply softmax to get probabilities
        utilities = np.array(utilities)
        exp_utilities = np.exp(utilities - np.max(utilities))  # numerical stability
        probs = exp_utilities / np.sum(exp_utilities)
        
        # Modes are ordered as: SAV, Taxi/solo-hailing, Metro, Bus
        sav_probabilities.append(probs[0])
        taxi_probabilities.append(probs[1])
        metro_probabilities.append(probs[2])
        bus_probabilities.append(probs[3])
    
    plt.plot(price_points, sav_probabilities, label='SAV', linewidth=2)
    plt.plot(price_points, taxi_probabilities, label='Taxi/solo-hailing', linewidth=2)
    plt.plot(price_points, metro_probabilities, label='Metro', linewidth=2)
    plt.plot(price_points, bus_probabilities, label='Bus', linewidth=2)
    plt.title('Price Sensitivity Analysis')
    plt.xlabel('Price (HKD)')
    plt.ylabel('Choice Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_sensitivity.png')
    plt.close()

    print("\nAll visualizations have been generated and saved as separate PNG files:")
    print("1. coefficient_plot.png")
    print("2. mode_choice_distribution.png")
    print("3. demographics_age.png")
    print("4. demographics_income.png")
    print("5. demographics_education.png")
    print("6. demographics_gender.png")
    print("7. cost_time_tradeoff.png")
    print("8. shared_mobility_experience.png")
    print("9. av_experience.png")
    print("10. price_sensitivity.png")


def calculate_statistics(encoded_data, feature_columns):
    stats_dict = {}
    
    # Mode choice statistics
    stats_dict['mode_choice'] = encoded_data[encoded_data['chosen']==1]['mode'].value_counts(normalize=True)
    
    # Get all encoded variable names (excluding constant and basic features for demographic stats)
    demographic_vars = [
        'Gender_encoded', 'Age_encoded', 'Income_encoded', 'Education_encoded',
        'Household_Size_encoded', 'Car_Ownership_encoded'
    ]
    transport_vars = [
        'PT_Accessibility_encoded', 'Shared_Mobility_Frequency_encoded',
        'AV_Experience_encoded', 'SM_Experience_encoded', 'SM_Willingness_encoded'
    ]
    categorical_vars = ['Major_encoded', 'Region_encoded']
    all_demographic_vars = demographic_vars + transport_vars + categorical_vars
    
    # Average characteristics of SAV choosers - ALL variables
    sav_choosers = encoded_data[
        (encoded_data['mode']=='SAV') & (encoded_data['chosen']==1)
    ]
    
    stats_dict['sav_choosers'] = {}
    for var in all_demographic_vars:
        if var in encoded_data.columns:
            stats_dict['sav_choosers'][f'avg_{var}'] = sav_choosers[var].mean()
    
    # Add percentage calculations for binary variables
    if 'Gender_encoded' in encoded_data.columns:
        stats_dict['sav_choosers']['pct_male'] = (sav_choosers['Gender_encoded']==0).mean()
    if 'Car_Ownership_encoded' in encoded_data.columns:
        stats_dict['sav_choosers']['pct_car_owners'] = sav_choosers['Car_Ownership_encoded'].mean()
    
    # Mode-specific statistics - ALL modes
    stats_dict['mode_characteristics'] = {}
    modes = ['SAV', 'Taxi/solo-hailing', 'Metro', 'Bus']
    for mode in modes:
        mode_choosers = encoded_data[
            (encoded_data['mode']==mode) & (encoded_data['chosen']==1)
        ]
        if len(mode_choosers) > 0:
            stats_dict['mode_characteristics'][mode] = {}
            for var in all_demographic_vars:
                if var in encoded_data.columns:
                    stats_dict['mode_characteristics'][mode][f'avg_{var}'] = mode_choosers[var].mean()
    
    # Overall sample statistics - ALL variables
    chosen_data = encoded_data[encoded_data['chosen']==1]
    stats_dict['sample_characteristics'] = {}
    for var in all_demographic_vars:
        if var in encoded_data.columns:
            stats_dict['sample_characteristics'][f'avg_{var}'] = chosen_data[var].mean()
    
    # Correlation matrix for ALL variables
    key_vars = ['cost', 'travel_time', 'waiting_time'] + all_demographic_vars
    key_vars = [v for v in key_vars if v in encoded_data.columns]
    stats_dict['correlations'] = encoded_data[key_vars].corr()
    
    return stats_dict


def plot_correlation_heatmap(stats_dict, figsize=(14, 12), cmap='coolwarm',
                             annot=True, save_path=None, title='Correlation Heatmap of Variables',
                             mask_upper=False):

    corr = stats_dict['correlations'].copy()

    # Clean up labels (remove the "_encoded" suffix for readability)
    pretty_labels = [c.replace('_encoded', '').replace('_', ' ').title()
                     for c in corr.columns]
    corr.columns = pretty_labels
    corr.index = pretty_labels

    # Optional mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt='.2f',
        annot_kws={'size': 8},
        vmin=-1, vmax=1, center=0,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.75, 'label': 'Pearson Correlation'},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")

    # plt.show()
    return fig, ax

def format_model_equation(feature_columns, weights, decimals=4):
    """
    Build human-readable strings for:
      - the generic utility V = w1*x1 + w2*x2 + ...
      - the per-alternative utility V_SAV, V_Taxi, V_Metro, V_Bus
      - the softmax choice probability
    Returns a single multi-line string ready to write to a file.
    """
    w = dict(zip(feature_columns, weights))

    def term(coef, varname):
        sign = '+' if coef >= 0 else '-'
        return f" {sign} {abs(coef):.{decimals}f}*{varname}"

    # ---- adaptive formatter for the coefficient table ----
    def fmt_coef(val):
        """Use scientific notation for very small magnitudes so small
        coefficients (e.g. on Income, which has values in the thousands)
        don't display as 0.0000."""
        a = abs(val)
        if a == 0:
            return f"{val:+.{decimals}f}"
        if a < 10 ** (-decimals):          # would round to 0 at fixed decimals
            return f"{val:+.6e}"           # e.g. +3.214567e-06
        if a < 1e-2:
            return f"{val:+.8f}"           # extra digits for small but visible
        return f"{val:+.{decimals}f}"

    # ---- 1. Generic utility (all features) ----
    generic_terms = ''.join(term(w[v], v) for v in feature_columns).lstrip(' +')
    generic_eq = f"V = {generic_terms}"

    # ---- 2. Per-alternative utility ----
    los_vars = ['cost', 'travel_time', 'waiting_time']

    def utility_for_alt(alt_label, asc_name, interaction_suffix):
        parts = []
        if asc_name is not None and asc_name in w:
            parts.append(f"{w[asc_name]:+.{decimals}f}")
        for v in los_vars:
            parts.append(term(w[v], v).strip())
        if interaction_suffix is not None:
            for col in feature_columns:
                if col.endswith(interaction_suffix):
                    demo_name = col.split('_x_')[0]
                    parts.append(term(w[col], demo_name).strip())
        eq = parts[0]
        for p in parts[1:]:
            if p.startswith(('+', '-')):
                eq += ' ' + p[0] + ' ' + p[1:]
            else:
                eq += ' + ' + p
        return f"V_{alt_label} = {eq}"

    v_sav   = utility_for_alt('SAV',   'ASC_SAV',   '_x_SAV')
    v_taxi  = utility_for_alt('Taxi',  'ASC_Taxi',  '_x_Taxi')
    v_metro = utility_for_alt('Metro', 'ASC_Metro', '_x_Metro')
    v_bus   = utility_for_alt('Bus',   None,        None)

    # ---- 3. Coefficient table (adaptive precision) ----
    coef_lines = ["Coefficient table (variable : weight):"]
    name_w = max(len(c) for c in feature_columns)
    for c in feature_columns:
        coef_lines.append(f"  {c.ljust(name_w)} : {fmt_coef(w[c])}")

    # ---- 4. Softmax probability ----
    prob_eq = (
        "Choice probability (Multinomial Logit):\n"
        "  P(i | individual) = exp(V_i) / [exp(V_SAV) + exp(V_Taxi) "
        "+ exp(V_Metro) + exp(V_Bus)]"
    )

    sections = [
        "=" * 80,
        "ESTIMATED MODEL",
        "=" * 80,
        "",
        "Generic utility specification (all model features stacked):",
        generic_eq,
        "",
        "Alternative-specific utilities (Bus = reference, no ASC, no demographics):",
        v_sav,
        v_taxi,
        v_metro,
        v_bus,
        "",
        prob_eq,
        "",
        *coef_lines,
        "=" * 80,
        "",
    ]
    return "\n".join(sections)


def plot_sav_probability_individual(weights, feature_columns, scaler, scenarios,
                                    encoded_data, encoding_mappings,
                                    save_dir='sav_plots', n_points=50):
    """
    Create one individual figure per demographic variable showing P(SAV).
    Axis ticks/labels are mapped back to original category names via
    encoding_mappings: {encoded_col_name: {int: original_label}}.
    """
    os.makedirs(save_dir, exist_ok=True)

    modes = ['SAV', 'Taxi/solo-hailing', 'Metro', 'Bus']
    asc_map = {'SAV': 'ASC_SAV', 'Taxi/solo-hailing': 'ASC_Taxi',
               'Metro': 'ASC_Metro', 'Bus': None}

    # Mean LOS values per mode
    los = {m: {k: np.mean([scenarios[s][m][k] for s in scenarios])
               for k in ['cost', 'travel_time', 'waiting_time']} for m in modes}

    # Recover the demographic vars actually used in the model
    individual_vars = list(dict.fromkeys(
        c.split('_x_')[0] for c in feature_columns if '_x_SAV' in c
    ))
    sample_means = {v: encoded_data[v].mean() for v in individual_vars}
    n_features = len(feature_columns)

    def build_row(mode, demo_overrides=None):
        row = np.zeros(n_features)
        if asc_map[mode] is not None:
            row[feature_columns.index(asc_map[mode])] = 1.0
        cost_v = (los[mode]['cost']         - scaler.mean_[0]) / scaler.scale_[0]
        tt_v   = (los[mode]['travel_time']  - scaler.mean_[1]) / scaler.scale_[1]
        wt_v   = (los[mode]['waiting_time'] - scaler.mean_[2]) / scaler.scale_[2]
        row[feature_columns.index('cost')]         = cost_v
        row[feature_columns.index('travel_time')]  = tt_v
        row[feature_columns.index('waiting_time')] = wt_v
        if mode != 'Bus':
            alt = {'SAV': 'SAV', 'Taxi/solo-hailing': 'Taxi', 'Metro': 'Metro'}[mode]
            for v in individual_vars:
                val = (sample_means[v] if (demo_overrides is None or v not in demo_overrides)
                       else demo_overrides[v])
                col = f'{v}_x_{alt}'
                if col in feature_columns:
                    row[feature_columns.index(col)] = val
        return row

    def softmax_p(rows):
        u = rows @ weights
        u = u - u.max()
        e = np.exp(u)
        return e / e.sum()

    saved_paths = []

    for var in individual_vars:
        unique_vals = np.sort(np.unique(encoded_data[var].dropna().values))
        mapping = encoding_mappings.get(var, {})

        # Decide whether to treat as categorical (bar) or continuous (line)
        is_categorical = (len(unique_vals) <= 15) or bool(mapping)

        if is_categorical:
            values = unique_vals
        else:
            values = np.linspace(unique_vals.min(), unique_vals.max(), n_points)

        sav_probs = []
        for v in values:
            rows = np.vstack([build_row(m, {var: v}) for m in modes])
            sav_probs.append(softmax_p(rows)[0])

        # Pretty original variable name (strip "_encoded", replace underscores)
        original_name = var.replace('_encoded', '').replace('_', ' ')

        fig, ax = plt.subplots(figsize=(8, 5))

        is_categorical = (len(unique_vals) <= 25) or bool(mapping)

        if is_categorical:
            ax.bar(range(len(values)), sav_probs,
                color='#3498db', alpha=0.85, edgecolor='white')
            tick_labels = [_decode(v, mapping) for v in values]
            ax.set_xticks(range(len(values)))
            long_labels = max(len(t) for t in tick_labels) > 6
            ax.set_xticklabels(
                tick_labels,
                rotation=30 if long_labels else 0,
                ha='right' if long_labels else 'center'
            )
            for i, p in enumerate(sav_probs):
                ax.text(i, p + 0.005, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.plot(values, sav_probs, color='#3498db', linewidth=2.2)
            ax.fill_between(values, 0, sav_probs, color='#3498db', alpha=0.15)

        ax.set_title(f'Probability of Choosing SAV by {original_name}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel(original_name, fontsize=11)
        ax.set_ylabel('P(SAV)', fontsize=11)
        ax.set_ylim(0, max(0.6, max(sav_probs) * 1.20))
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(save_dir, f'P_SAV_vs_{var.replace("_encoded","")}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig)
        saved_paths.append(out_path)

    print(f"Saved {len(saved_paths)} individual plots to '{save_dir}/'")
    return saved_paths

# Run the analysis
if __name__ == "__main__":
    # Prepare data
    X, y, encoded_data, scaler, feature_columns, group_ids, inverse_mappings = \
        prepare_data_for_analysis()

    weights, X_test, y_test = train_and_evaluate_model(X, y, feature_columns, group_ids)

    # ============================================================
    # MODEL VALIDATION
    # ============================================================
    print("\nCalculating model validation metrics...")
    
    # Calculate standard errors, t-values, and p-values
    std_errors, t_values, p_values = calculate_standard_errors(weights, X, y, group_ids)
    
    # Calculate goodness of fit metrics
    goodness_of_fit = calculate_goodness_of_fit(weights, X, y, group_ids)
    
    # Print validation summary
    print_validation_summary(weights, feature_columns, std_errors, t_values, p_values, goodness_of_fit)
    
    # Create validation visualizations
    create_validation_visualizations(weights, feature_columns, X, y, group_ids,
                                     std_errors, t_values, p_values, goodness_of_fit)

    plot_sav_probability_individual(
        weights=weights,
        feature_columns=feature_columns,
        scaler=scaler,
        scenarios=scenarios,
        encoded_data=encoded_data,
        encoding_mappings=inverse_mappings,   # <-- inverted, with _encoded keys
        save_dir='sav_plots'
    )

    # Print summary of variable effects
    print("\nVariable Effects Summary:")
    for name, weight in zip(feature_columns, weights):
        # if abs(weight) > 0.1:  # Show only significant effects
        effect = "positive" if weight > 0 else "negative"
        print(f"{name}: {effect} effect (coefficient = {weight:.4f})")
    
    # Generate visualizations
    create_visualizations(encoded_data, weights, feature_columns, scaler, X, scenarios)
    
    # Calculate and print statistics
    statistics = calculate_statistics(encoded_data, feature_columns)
    plot_correlation_heatmap(statistics, save_path='correlation_heatmap.png', mask_upper=True)
    
    model_text = format_model_equation(feature_columns, weights, decimals=4)

    with open('statistics_summary.txt', 'w', encoding='utf-8') as f:
        f.write(model_text)                       # <-- model equation first
        f.write("\nMode Choice Distribution:\n")
        f.write(statistics['mode_choice'].to_string())
        f.write("\n\nSAV Chooser Characteristics:\n")
        for key, value in statistics['sav_choosers'].items():
            f.write(f"{key}: {value:.2f}\n")
        f.write("\n\nMode Characteristics (All Modes):\n")
        for mode, chars in statistics['mode_characteristics'].items():
            f.write(f"\n{mode}:\n")
            for key, value in chars.items():
                f.write(f"  {key}: {value:.2f}\n")
        f.write("\n\nSample Characteristics (Overall):\n")
        for key, value in statistics['sample_characteristics'].items():
            f.write(f"{key}: {value:.2f}\n")
        f.write("\nKey Variable Correlations:\n")
        f.write(statistics['correlations'].to_string())
    # print("\nMode Choice Distribution:")
    # print(statistics['mode_choice'])
    
    # print("\nSAV Chooser Characteristics:")
    # for key, value in statistics['sav_choosers'].items():
    #     print(f"{key}: {value:.2f}")
    
    # print("\nKey Variable Correlations:")
    # print(statistics['correlations'])
