import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

file_path = r"C:\Users\jimng\OneDrive - The University of Hong Kong\Desktop\School\HKU Course\Dissertation\Survey\Survey Data.xlsx" 
df = pd.read_excel(file_path, sheet_name="Data", usecols="A:W")

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

def process_scenario_responses(df, scenarios):
    processed_data = []
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
                for col in df.columns:
                    if col not in ['Scenario_4.1', 'Scenario_4.2', 'Scenario_4.3', 'Scenario_4.4']:
                        data_point[col] = row[col]
                processed_data.append(data_point)
    result_df = pd.DataFrame(processed_data)
    return result_df

def encode_all_variables(df):
    encoded_df = df.copy()
    for col, mapping in ENCODING_MAPPINGS.items():
        if col in encoded_df.columns:
            unique_values = encoded_df[col].unique()
            if len(unique_values) == 1:
                encoded_df[f'{col}_encoded'] = 1
            else:
                encoded_df[f'{col}_encoded'] = encoded_df[col].map(mapping)
            missing = encoded_df[f'{col}_encoded'].isna().sum()
    categorical_cols = ['Major', 'Region']
    label_encoders = {}
    for col in categorical_cols:
        if col in encoded_df.columns:
            label_encoders[col] = LabelEncoder()
            encoded_df[f'{col}_encoded'] = label_encoders[col].fit_transform(encoded_df[col])
            missing = encoded_df[f'{col}_encoded'].isna().sum()
    return encoded_df

def prepare_data_for_analysis():
    scenario_data = process_scenario_responses(df, scenarios)
    encoded_data = encode_all_variables(scenario_data)
    basic_features = ['cost', 'travel_time', 'waiting_time']
    demographic_features = [
        'Gender_encoded',
        'Age_encoded',
        'Income_encoded',
        'Education_encoded',
        'Household_Size_encoded',
        'Car_Ownership_encoded'
    ]
    transport_features = [
        'PT_Accessibility_encoded',
        'Shared_Mobility_Frequency_encoded',
        'AV_Experience_encoded',
        'SM_Experience_encoded',
        'SM_Willingness_encoded'
    ]
    categorical_features = [
        'Major_encoded',
        'Region_encoded'
    ]
    feature_columns = basic_features + demographic_features + transport_features + categorical_features
    encoded_data['constant'] = 1
    feature_columns = ['constant'] + feature_columns
    encoded_data = encoded_data.dropna(subset=feature_columns)
    if encoded_data.shape[0] == 0:
        raise ValueError("All rows were dropped due to missing values!")
    X = encoded_data[feature_columns].values
    y = encoded_data['chosen'].values
    scaler = StandardScaler()
    X_scaled = np.column_stack([
        X[:, 0],
        scaler.fit_transform(X[:, 1:])
    ])
    return X_scaled, y, encoded_data, scaler, feature_columns

def train_and_evaluate_model(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    initial_weights = np.zeros(X.shape[1])
    result = optimize.minimize(
        fun=log_likelihood,
        x0=initial_weights,
        args=(X_train, y_train),
        method='BFGS',
        options={'maxiter': 1000}
    )
    return result.x, X_test, y_test

def log_likelihood(weights, X, y):
    utilities = np.dot(X, weights)
    exp_utilities = np.exp(utilities)
    probabilities = exp_utilities / np.sum(exp_utilities)
    log_likelihood_value = np.sum(y * np.log(probabilities + 1e-10))
    return -log_likelihood_value

def create_visualizations(encoded_data, weights, feature_columns, scaler, X_scaled):
    plt.style.use('default')
    print("All coefficients:")
    for name, weight in zip(feature_columns, weights):
        print(f"{name}: {weight:.4f}")
    threshold = 0.01
    significant_features = [(name, weight) for name, weight in zip(feature_columns, weights) 
                          if abs(weight) > threshold]
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


    chosen_data = encoded_data[encoded_data['chosen'] == 1]
    plt.figure(figsize=(10, 6))
    age_means = chosen_data.groupby('mode')['Age_encoded'].mean()
    age_means.plot(kind='bar')
    plt.title('Average Age by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.savefig('demographics_age.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    income_means = chosen_data.groupby('mode')['Income_encoded'].mean()
    income_means.plot(kind='bar')
    plt.title('Average Income by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Income')
    plt.tight_layout()
    plt.savefig('demographics_income.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    edu_means = chosen_data.groupby('mode')['Education_encoded'].mean()
    edu_means.plot(kind='bar')
    plt.title('Average Education Level by Chosen Mode')
    plt.xlabel('Transport Mode')
    plt.ylabel('Education Level')
    plt.tight_layout()
    plt.savefig('demographics_education.png')
    plt.close()

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


    plt.figure(figsize=(10, 6))
    price_points = np.linspace(0, 500, 50)
    probabilities = []
    for price in price_points:
        X_temp = X_scaled.copy()
        X_temp[:, 1] = (price - scaler.mean_[0]) / scaler.scale_[0]
        utilities = np.dot(X_temp, weights)
        exp_utilities = np.exp(utilities)
        prob = exp_utilities / np.sum(exp_utilities)
        probabilities.append(np.mean(prob))
    plt.plot(price_points, probabilities)
    plt.title('Price Sensitivity Analysis')
    plt.xlabel('Price')
    plt.ylabel('Choice Probability')
    plt.tight_layout()
    plt.savefig('price_sensitivity.png')
    plt.close()



def calculate_statistics(encoded_data):
    stats_dict = {}
    stats_dict['mode_choice'] = encoded_data[encoded_data['chosen']==1]['mode'].value_counts(normalize=True)
    sav_choosers = encoded_data[
        (encoded_data['mode']=='SAV') & (encoded_data['chosen']==1)
    ]
    stats_dict['sav_choosers'] = {
        'avg_age': sav_choosers['Age_encoded'].mean(),
        'avg_income': sav_choosers['Income_encoded'].mean(),
        'avg_education': sav_choosers['Education_encoded'].mean(),
        'pct_male': (sav_choosers['Gender_encoded']==0).mean(),
        'pct_car_owners': sav_choosers['Car_Ownership_encoded'].mean()
    }
    key_vars = ['cost', 'travel_time', 'waiting_time', 'Age_encoded', 
                'Income_encoded', 'Education_encoded']
    stats_dict['correlations'] = encoded_data[key_vars].corr()
    return stats_dict

X_scaled, y, encoded_data, scaler, feature_columns = prepare_data_for_analysis()
weights, X_test, y_test = train_and_evaluate_model(X_scaled, y, feature_columns)
print("\nVariable Effects Summary:")
for name, weight in zip(feature_columns, weights):
    if abs(weight) > 0.1:
        effect = "positive" if weight > 0 else "negative"
        print(f"{name}: {effect} effect (coefficient = {weight:.4f})")
create_visualizations(encoded_data, weights, feature_columns, scaler, X_scaled)
statistics = calculate_statistics(encoded_data)
print("\nMode Choice Distribution:")
print(statistics['mode_choice'])
print("\nSAV Chooser Characteristics:")
for key, value in statistics['sav_choosers'].items():
    print(f"{key}: {value:.2f}")
print("\nKey Variable Correlations:")
print(statistics['correlations'])
