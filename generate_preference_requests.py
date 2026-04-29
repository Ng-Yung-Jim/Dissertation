import pandas as pd
import numpy as np
import random
import math
import osmnx as ox
import json
import os
from scipy import stats
import matplotlib.pyplot as plt

# Hong Kong bounding box
MAXLAT = 22.6
MINLAT = 22.1
MAXLNG = 114.5
MINLNG = 113.8

# Simulation time range (in seconds, 6:00 to 10:00)
START_TIME = 21600
END_TIME = 36000

np.random.seed(0)

# ============================================================================
# SURVEY-DERIVED PARAMETERS
# ============================================================================

# Scenario parameters from survey (for trip distance/time distributions)
SCENARIOS = {
    'Scenario_4.1': {  # Long-distance
        'SAV': {'cost': 240, 'travel_time': 40, 'waiting_time': 10},
        'Taxi': {'cost': 300, 'travel_time': 35, 'waiting_time': 5},
        'Metro': {'cost': 110, 'travel_time': 50, 'waiting_time': 15},
        'Bus': {'cost': 50, 'travel_time': 75, 'waiting_time': 15}
    },
    'Scenario_4.2': {  # Medium-distance
        'SAV': {'cost': 50, 'travel_time': 35, 'waiting_time': 3},
        'Taxi': {'cost': 90, 'travel_time': 25, 'waiting_time': 5},
        'Metro': {'cost': 15, 'travel_time': 35, 'waiting_time': 10},
        'Bus': {'cost': 12, 'travel_time': 40, 'waiting_time': 15}
    },
    'Scenario_4.3': {  # Short-distance
        'SAV': {'cost': 20, 'travel_time': 7, 'waiting_time': 3},
        'Taxi': {'cost': 40, 'travel_time': 5, 'waiting_time': 2},
        'Metro': {'cost': 5, 'travel_time': 10, 'waiting_time': 10},
        'Bus': {'cost': 7, 'travel_time': 15, 'waiting_time': 5}
    },
    'Scenario_4.4': {  # Medium-distance
        'SAV': {'cost': 40, 'travel_time': 16, 'waiting_time': 5},
        'Taxi': {'cost': 50, 'travel_time': 10, 'waiting_time': 4},
        'Metro': {'cost': 10, 'travel_time': 10, 'waiting_time': 10},
        'Bus': {'cost': 5, 'travel_time': 15, 'waiting_time': 15}
    }
}

# Extract trip distance/time distributions from scenarios
# Assuming average speed of 15 m/s for converting time to distance
AVG_SPEED_MPS = 15

# Trip distance ranges by scenario (in meters) - derived from travel_time * speed
SCENARIO_TRIP_DISTANCES = {
    'Scenario_4.1': {
        'SAV': 40 * 60 * AVG_SPEED_MPS,      # 40 min -> meters
        'Taxi': 35 * 60 * AVG_SPEED_MPS,
        'Metro': 50 * 60 * AVG_SPEED_MPS,
        'Bus': 75 * 60 * AVG_SPEED_MPS
    },
    'Scenario_4.2': {
        'SAV': 35 * 60 * AVG_SPEED_MPS,
        'Taxi': 25 * 60 * AVG_SPEED_MPS,
        'Metro': 35 * 60 * AVG_SPEED_MPS,
        'Bus': 40 * 60 * AVG_SPEED_MPS
    },
    'Scenario_4.3': {
        'SAV': 7 * 60 * AVG_SPEED_MPS,
        'Taxi': 5 * 60 * AVG_SPEED_MPS,
        'Metro': 10 * 60 * AVG_SPEED_MPS,
        'Bus': 15 * 60 * AVG_SPEED_MPS
    },
    'Scenario_4.4': {
        'SAV': 16 * 60 * AVG_SPEED_MPS,
        'Taxi': 10 * 60 * AVG_SPEED_MPS,
        'Metro': 10 * 60 * AVG_SPEED_MPS,
        'Bus': 15 * 60 * AVG_SPEED_MPS
    }
}

# Mode share weights from survey (approximate - will be updated when coefficients are available)
# These represent the probability of each scenario being selected
SCENARIO_WEIGHTS = {
    'Scenario_4.1': 0.25,
    'Scenario_4.2': 0.25,
    'Scenario_4.3': 0.25,
    'Scenario_4.4': 0.25
}

# ============================================================================
# LOGIT MODEL COEFFICIENTS (from estimated model in statistics_summary.txt)
# ============================================================================

# Actual coefficients from the estimated multinomial logit model
# Note: For utility calculation, cost should be negative (higher cost = lower utility)
# The model was estimated with positive cost coefficient, but we invert for proper utility
DEFAULT_COEFFICIENTS = {
    # Generic coefficients (common across all alternatives)
    # Using negative cost for proper utility (higher cost = lower utility)
    'cost': -0.7371,        # Negative: higher cost -> lower utility
    'travel_time': 1.1209,  # Positive: shorter travel time -> higher utility
    'waiting_time': 0.3063, # Positive: shorter wait -> higher utility
    
    # Alternative-specific constants (ASC)
    'ASC_SAV': 1.5031,
    'ASC_Taxi': -0.6586,
    'ASC_Metro': 1.5795,
    # Bus is reference (ASC = 0)
    
    # Demographic interaction coefficients
    'Gender_encoded_x_SAV': -0.3873,
    'Gender_encoded_x_Taxi': 0.0598,
    'Gender_encoded_x_Metro': 0.2290,
    
    'Age_encoded_x_SAV': -0.0153,
    'Age_encoded_x_Taxi': 0.0148,
    'Age_encoded_x_Metro': 0.0052,
    
    'Income_encoded_x_SAV': 0.00002395,
    'Income_encoded_x_Taxi': 0.00003046,
    'Income_encoded_x_Metro': 0.000003578,
    
    'Education_encoded_x_SAV': 0.2020,
    'Education_encoded_x_Taxi': 0.1353,
    'Education_encoded_x_Metro': 0.1523,
    
    'Household_Size_encoded_x_SAV': -0.5806,
    'Household_Size_encoded_x_Taxi': -0.2453,
    'Household_Size_encoded_x_Metro': -0.3744,
    
    'Car_Ownership_encoded_x_SAV': -0.2303,
    'Car_Ownership_encoded_x_Taxi': -0.0526,
    'Car_Ownership_encoded_x_Metro': -0.5039,
    
    'PT_Accessibility_encoded_x_SAV': 0.3476,
    'PT_Accessibility_encoded_x_Taxi': 0.1788,
    'PT_Accessibility_encoded_x_Metro': 0.2251,
    
    'Shared_Mobility_Frequency_encoded_x_SAV': 0.2553,
    'Shared_Mobility_Frequency_encoded_x_Taxi': 0.0433,
    'Shared_Mobility_Frequency_encoded_x_Metro': 0.1812,
    
    'AV_Experience_encoded_x_SAV': 0.2917,
    'AV_Experience_encoded_x_Taxi': 0.4709,
    'AV_Experience_encoded_x_Metro': 0.1741,
    
    'SM_Experience_encoded_x_SAV': 0.1800,
    'SM_Experience_encoded_x_Taxi': -0.0828,
    'SM_Experience_encoded_x_Metro': -0.3150,
    
    'SM_Willingness_encoded_x_SAV': -0.3657,
    'SM_Willingness_encoded_x_Taxi': -0.4804,
    'SM_Willingness_encoded_x_Metro': -0.4598,
    
    'Major_encoded_x_SAV': -0.1549,
    'Major_encoded_x_Taxi': -0.0451,
    'Major_encoded_x_Metro': -0.1133,
    
    'Region_encoded_x_SAV': -0.0014,
    'Region_encoded_x_Taxi': 0.0041,
    'Region_encoded_x_Metro': -0.0089
}

# Willingness to pay for time savings (HKD per unit of cost)
# WTP = beta_time / |beta_cost|
DEFAULT_WTP_PER_MINUTE = DEFAULT_COEFFICIENTS['travel_time'] / abs(DEFAULT_COEFFICIENTS['cost'])  # ≈ 1.52

# ============================================================================
# DEMOGRAPHIC ENCODINGS (numeric values for model inference)
# ============================================================================

# Mapping from categorical values to numeric encoded values
DEMOGRAPHIC_ENCODINGS = {
    'Gender': {
        'Female': 0,
        'Male': 1
    },
    'Age': {
        'Under 18': 16,
        '18-24': 21,
        '25-34': 29,
        '35-44': 39,
        '45-54': 49,
        '55-64': 59,
        '65 and above': 68
    },
    'Income': {
        'Less than $10400': 5200,
        '$10400–$14300': 12350,
        '$14300–$19800': 17050,
        '$19800–$31100': 25450,
        '$31100–$48500': 39800,
        '$48500 or more': 60000,
        'Unemployed / Prefer not to say': 0
    },
    'Education': {
        'High school or below': 1,
        'Some college/Associate': 2,
        'Bachelor\'s degree': 3,
        'Graduate degree': 4
    },
    'Household_Size': {
        '1': 1, '2': 2, '3': 3, '4': 4, '5+': 5
    },
    'Car_Ownership': {
        'No': 0,
        'Yes': 1
    },
    'PT_Accessibility': {
        'Low': 1,
        'Medium': 2,
        'High': 3
    },
    'Shared_Mobility_Frequency': {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Often': 3,
        'Very often': 4
    },
    'AV_Experience': {
        'No': 0,
        'Yes': 1
    },
    'SM_Experience': {
        'No': 0,
        'Yes': 1
    },
    'SM_Willingness': {
        'No': 0,
        'Yes': 1
    },
    'Major': {
        'Engineering/Architecture': 1,
        'Business/Finance': 2,
        'Social Sciences': 3,
        'Science/Technology': 4,
        'Arts/Humanities': 5,
        'Other': 6
    },
    'Region': {
        'Hong Kong Island': 1,
        'Kowloon': 2,
        'New Territories East': 3,
        'New Territories West': 4,
        'Islands': 5,
        'Sai Kung': 6,
        'Sha Tin': 7,
        'Tuen Mun': 8,
        'Tsuen Wan': 9,
        'Other': 10
    }
}


# ============================================================================
# DEMOGRAPHIC DISTRIBUTIONS (from survey statistics)
# ============================================================================

# These distributions match the statistics_summary.txt sample characteristics
DEMOGRAPHIC_DISTRIBUTIONS = {
    'Gender': {
        'Female': 0.55,
        'Male': 0.45
    },
    'Age': {
        'Under 18': 0.05,
        '18-24': 0.25,
        '25-34': 0.30,
        '35-44': 0.20,
        '45-54': 0.12,
        '55-64': 0.05,
        '65 and above': 0.03
    },
    'Income': {
        'Less than $10400': 0.15,
        '$10400–$14300': 0.12,
        '$14300–$19800': 0.18,
        '$19800–$31100': 0.25,
        '$31100–$48500': 0.20,
        '$48500 or more': 0.08,
        'Unemployed / Prefer not to say': 0.02
    },
    'Education': {
        'High school or below': 0.20,
        'Some college/Associate': 0.25,
        'Bachelor\'s degree': 0.40,
        'Graduate degree': 0.15
    },
    'Household_Size': {
        '1': 0.15,
        '2': 0.25,
        '3': 0.30,
        '4': 0.20,
        '5+': 0.10
    },
    'Car_Ownership': {
        'No': 0.28,
        'Yes': 0.72
    },
    'PT_Accessibility': {
        'Low': 0.20,
        'Medium': 0.45,
        'High': 0.35
    },
    'Shared_Mobility_Frequency': {
        'Never': 0.25,
        'Rarely': 0.30,
        'Sometimes': 0.25,
        'Often': 0.15,
        'Very often': 0.05
    },
    'AV_Experience': {
        'No': 0.29,
        'Yes': 0.71
    },
    'SM_Experience': {
        'No': 0.64,
        'Yes': 0.36
    },
    'SM_Willingness': {
        'No': 0.17,
        'Yes': 0.83
    },
    'Major': {
        'Engineering/Architecture': 0.25,
        'Business/Finance': 0.20,
        'Social Sciences': 0.15,
        'Science/Technology': 0.25,
        'Arts/Humanities': 0.10,
        'Other': 0.05
    },
    'Region': {
        'Hong Kong Island': 0.20,
        'Kowloon': 0.25,
        'New Territories East': 0.15,
        'New Territories West': 0.10,
        'Islands': 0.03,
        'Sai Kung': 0.05,
        'Sha Tin': 0.10,
        'Tuen Mun': 0.05,
        'Tsuen Wan': 0.05,
        'Other': 0.02
    }
}

# Target mode shares from statistics
TARGET_MODE_SHARES = {
    'SAV': 0.314,
    'Taxi': 0.164,
    'Metro': 0.298,
    'Bus': 0.224
}


def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two coordinates in meters"""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c * 1000  # Return in meters


def load_network_nodes(graphml_path='./data/hongkong.graphml'):
    """Load all node coordinates from the road network"""
    print(f"Loading road network from {graphml_path}...")
    try:
        G = ox.load_graphml(graphml_path)
        nodes = []
        for node_id, node_data in G.nodes(data=True):
            lat = node_data.get('y')
            lng = node_data.get('x')
            if lat is not None and lng is not None:
                nodes.append((lng, lat))  # (longitude, latitude) format
        print(f"Loaded {len(nodes)} nodes from network")
        return nodes
    except Exception as e:
        print(f"Error loading network: {e}")
        print("Fallback: generating random coordinates within bounding box")
        return None


def sample_trip_from_survey():
    """
    Sample trip characteristics from survey scenario distributions.
    Returns: (trip_distance, expected_fare, scenario_name)
    """
    # Select scenario based on weights
    scenario_name = random.choices(
        list(SCENARIO_WEIGHTS.keys()),
        weights=list(SCENARIO_WEIGHTS.values())
    )[0]
    
    # Select mode within scenario (for fare calculation)
    mode = random.choice(['SAV', 'Taxi', 'Metro', 'Bus'])
    
    # Get trip distance for this mode in this scenario
    trip_distance = SCENARIO_TRIP_DISTANCES[scenario_name][mode]
    
    # Add some randomness (+/- 20%)
    trip_distance *= random.uniform(0.8, 1.2)
    
    # Calculate expected fare based on scenario cost
    # This is the fare that makes this mode competitive
    scenario_cost = SCENARIOS[scenario_name][mode]['cost']
    
    # Add fare variation based on distance
    # Base fare + distance-based component
    base_fare = scenario_cost * 0.5
    distance_fare = (trip_distance / 1000) * 9.5  # ~9.5 HKD per km
    expected_fare = (base_fare + distance_fare) / 2
    
    # Add randomness to fare (+/- 15%)
    expected_fare *= random.uniform(0.85, 1.15)
    
    return trip_distance, max(10, expected_fare), scenario_name


def calculate_sav_fare(trip_distance_m, coefficients=None):
    """
    Calculate SAV fare based on willingness-to-pay from logit model.
    
    The fare is set to be competitive based on:
    - Trip distance
    - User's willingness to pay (from coefficients)
    - Competitor fares (Metro/Bus)
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    
    trip_distance_km = trip_distance_m / 1000
    
    # Reference fares (from survey scenarios)
    metro_fare = 5 + trip_distance_km * 2  # ~2 HKD/km
    bus_fare = 4 + trip_distance_km * 2     # ~2 HKD/km
    
    # SAV fare should be between Metro and Taxi
    # Based on WTP, users are willing to pay more for time savings
    wtp = -coefficients['travel_time'] / coefficients['cost']  # HKD per unit
    
    # Calculate fare that gives competitive utility
    # Higher WTP -> higher acceptable fare
    time_savings_min = int(random.uniform(-5, 5)) 
    premium_for_time = wtp * time_savings_min
    
    # Base SAV fare: Metro fare + premium for time savings
    sav_fare = bus_fare + premium_for_time * 0.3
    
    # Add distance-based component
    sav_fare += trip_distance_km * 5  # ~5 HKD/km
    
    # Ensure fare is reasonable (between bus and taxi)
    if trip_distance_km <= 7:
        taxi_fare = 29 + max(0, trip_distance_km - 2) * 10.5  # ~10.5 HKD/km
    else:
        taxi_fare = 102.5 + (trip_distance_km - 7) * 7  # ~7 HKD/km after 7 km
    sav_fare = min(sav_fare, taxi_fare * 0.8)  # At most 80% of taxi fare
    sav_fare = max(sav_fare, bus_fare * 1.5)   # At least 1.5x bus fare
    
    return round(sav_fare, 2)


def sample_user_profile():
    """
    Sample a user profile based on demographic distributions from survey.
    Returns: dict with all demographic attributes (both categorical and encoded)
    """
    profile = {}
    
    # Sample each demographic attribute
    for attr, dist in DEMOGRAPHIC_DISTRIBUTIONS.items():
        categories = list(dist.keys())
        weights = list(dist.values())
        profile[attr] = random.choices(categories, weights=weights)[0]
    
    # Convert to encoded numeric values for model inference
    profile['Gender_encoded'] = DEMOGRAPHIC_ENCODINGS['Gender'][profile['Gender']]
    profile['Age_encoded'] = DEMOGRAPHIC_ENCODINGS['Age'][profile['Age']]
    profile['Income_encoded'] = DEMOGRAPHIC_ENCODINGS['Income'][profile['Income']]
    profile['Education_encoded'] = DEMOGRAPHIC_ENCODINGS['Education'][profile['Education']]
    profile['Household_Size_encoded'] = DEMOGRAPHIC_ENCODINGS['Household_Size'][profile['Household_Size']]
    profile['Car_Ownership_encoded'] = DEMOGRAPHIC_ENCODINGS['Car_Ownership'][profile['Car_Ownership']]
    profile['PT_Accessibility_encoded'] = DEMOGRAPHIC_ENCODINGS['PT_Accessibility'][profile['PT_Accessibility']]
    profile['Shared_Mobility_Frequency_encoded'] = DEMOGRAPHIC_ENCODINGS['Shared_Mobility_Frequency'][profile['Shared_Mobility_Frequency']]
    profile['AV_Experience_encoded'] = DEMOGRAPHIC_ENCODINGS['AV_Experience'][profile['AV_Experience']]
    profile['SM_Experience_encoded'] = DEMOGRAPHIC_ENCODINGS['SM_Experience'][profile['SM_Experience']]
    profile['SM_Willingness_encoded'] = DEMOGRAPHIC_ENCODINGS['SM_Willingness'][profile['SM_Willingness']]
    profile['Major_encoded'] = DEMOGRAPHIC_ENCODINGS['Major'][profile['Major']]
    profile['Region_encoded'] = DEMOGRAPHIC_ENCODINGS['Region'][profile['Region']]
    
    return profile


def calculate_utility(mode, trip_attrs, user_profile, coefficients):
    """
    Calculate utility V for a given mode based on logit model.
    
    Parameters:
    -----------
    mode : str
        Mode name ('SAV', 'Taxi', 'Metro', 'Bus')
    trip_attrs : dict
        Trip attributes: cost, travel_time, waiting_time
    user_profile : dict
        User demographic profile with encoded values
    coefficients : dict
        Logit model coefficients
    
    Returns:
    --------
    float
        Utility V for this mode
    """
    # Get ASC (alternative-specific constant)
    asc_key = f'ASC_{mode}'
    V = coefficients.get(asc_key, 0)  # Bus has no ASC (reference)
    
    # Generic coefficients (common across all alternatives)
    V += coefficients['cost'] * trip_attrs['cost']
    V += coefficients['travel_time'] * trip_attrs['travel_time']
    V += coefficients['waiting_time'] * trip_attrs['waiting_time']
    
    # Demographic interaction terms
    demographic_vars = [
        'Gender_encoded', 'Age_encoded', 'Income_encoded', 'Education_encoded',
        'Household_Size_encoded', 'Car_Ownership_encoded', 'PT_Accessibility_encoded',
        'Shared_Mobility_Frequency_encoded', 'AV_Experience_encoded', 
        'SM_Experience_encoded', 'SM_Willingness_encoded', 
        'Major_encoded', 'Region_encoded'
    ]
    
    for var in demographic_vars:
        coeff_key = f'{var}_x_{mode}'
        if coeff_key in coefficients:
            V += coefficients[coeff_key] * user_profile[var]
    
    return V


def sample_mode_choice(trip_attrs, user_profile, coefficients):
    """
    Sample mode choice based on logit probabilities from utility calculation.
    
    Parameters:
    -----------
    trip_attrs : dict
        Trip attributes: cost, travel_time, waiting_time for each mode
    user_profile : dict
        User demographic profile with encoded values
    coefficients : dict
        Logit model coefficients
    
    Returns:
    --------
    str
        Selected mode ('SAV', 'Taxi', 'Metro', 'Bus')
    """
    modes = ['SAV', 'Taxi', 'Metro', 'Bus']
    
    # Normalize trip attributes to reasonable scales for utility calculation
    # The coefficients are estimated per unit of the attribute, but scenario costs
    # are in HKD. We need to scale appropriately.
    # Using very small cost scaling to reduce cost impact relative to ASC and time
    scale_cost = 0.001   # Divide cost by 1000 to get per 1000 HKD units
    scale_time = 1/60    # Keep time in minutes
    
    # Calculate utility for each mode
    utilities = {}
    for mode in modes:
        # Get trip attributes for this specific mode and normalize
        mode_attrs = {
            'cost': trip_attrs[mode]['cost'] * scale_cost,  # Per 1000 HKD
            'travel_time': trip_attrs[mode]['travel_time'] * scale_time,  # Minutes
            'waiting_time': trip_attrs[mode]['waiting_time'] * scale_time  # Minutes
        }
        utilities[mode] = calculate_utility(mode, mode_attrs, user_profile, coefficients)
    
    # Calculate logit probabilities
    max_util = max(utilities.values())  # For numerical stability
    
    exp_utils = {}
    sum_exp_utils = 0
    for mode in modes:
        exp_utils[mode] = math.exp(utilities[mode] - max_util)
        sum_exp_utils += exp_utils[mode]
    
    probabilities = {}
    for mode in modes:
        probabilities[mode] = exp_utils[mode] / sum_exp_utils
    
    # Sample mode based on probabilities
    mode_list = list(probabilities.keys())
    prob_list = list(probabilities.values())
    selected_mode = random.choices(mode_list, weights=prob_list)[0]
    
    return selected_mode, probabilities


def generate_preference_requests(
    num_requests=10000,
    graphml_path='./data/hongkong.graphml',
    coefficients=None,
    output_path='./data/requests/preference_based_requests.csv'
):
    """
    Generate synthetic Hong Kong ride requests based on survey preferences.
    
    Parameters:
    -----------
    num_requests : int
        Number of requests to generate
    graphml_path : str
        Path to road network graphml file
    coefficients : dict
        Logit model coefficients for fare calculation
    output_path : str
        Path to save output CSV
    
    Returns:
    --------
    pd.DataFrame
        Generated requests
    """
    requests = []
    filtered_requests = []
    
    print(f"Generating {num_requests} preference-based requests...")
    print(f"Using coefficients: {coefficients or DEFAULT_COEFFICIENTS}")
    
    # Load network nodes
    nodes = load_network_nodes(graphml_path)
    
    if nodes is None or len(nodes) < 2:
        print("ERROR: Road network has fewer than 2 nodes.")
        return None
    
    for order_id in range(1, num_requests + 1):
        # Sample trip characteristics from survey
        trip_distance, expected_fare, scenario = sample_trip_from_survey()
        
        # Select random origin and destination from network nodes
        origin_lng, origin_lat = random.choice(nodes)
        dest_lng, dest_lat = random.choice(nodes)
        
        # Ensure origin and destination are different
        while (origin_lng, origin_lat) == (dest_lng, dest_lat):
            dest_lng, dest_lat = random.choice(nodes)
        
        # Adjust distance to match sampled trip distance
        # (re-select nodes if needed, or just use the sampled distance)
        actual_distance = haversine_distance(origin_lat, origin_lng, dest_lat, dest_lng)
        
        # If actual distance is too different, scale the trip distance
        if actual_distance > 0:
            distance_ratio = trip_distance / actual_distance
            if distance_ratio > 2 or distance_ratio < 0.5:
                # Distance mismatch too large, use actual distance
                trip_distance = actual_distance
                expected_fare = calculate_sav_fare(trip_distance, coefficients)
        
        # Random start time (weighted towards peak hours)
        # Peak hours: 7-9 AM and 5-7 PM
        hour = random.randint(6, 9)  # 6:00 to 10:00
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        start_time = hour * 3600 + minute * 60 + second
        
        # Estimate trip time
        trip_time = trip_distance / AVG_SPEED_MPS
        
        # Assign to grid cells (10x10 grid)
        origin_grid_id = int((origin_lng - MINLNG) / (MAXLNG - MINLNG) * 10) * 10 + \
                        int((origin_lat - MINLAT) / (MAXLAT - MINLAT) * 10)
        dest_grid_id = int((dest_lng - MINLNG) / (MAXLNG - MINLNG) * 10) * 10 + \
                      int((dest_lat - MINLAT) / (MAXLAT - MINLAT) * 10)
        
        # Generate itinerary
        origin_id = int(order_id * 100 + random.random() * 10000)
        dest_id = int(order_id * 100 + 5000 + random.random() * 10000)
        itinerary_node_list = f"[{origin_id}, {dest_id}]"
        itinerary_segment_dis_list = f"[{trip_distance/1000:.2f}]"  # in km
        
        # Sample user profile with all demographics
        user_profile = sample_user_profile()
        
        # Get trip attributes for all modes from the scenario
        trip_attrs = SCENARIOS[scenario]
        
        # Sample mode choice using logit model (Option B)
        selected_mode, mode_probs = sample_mode_choice(trip_attrs, user_profile, coefficients)
        
        # Get trip attributes for the selected mode
        selected_attrs = trip_attrs[selected_mode]
        
        # Calculate fare based on selected mode's attributes
        # Use the cost from the scenario for the selected mode
        designed_reward = calculate_sav_fare(trip_distance, coefficients)
        
        # Adjust fare based on user profile
        # Higher income -> less price sensitive -> higher acceptable fare
        if user_profile['Income_encoded'] > 40000:
            designed_reward *= 1.2  # High income: willing to pay more
        elif user_profile['Income_encoded'] < 10000:
            designed_reward *= 0.8  # Low income: more price sensitive
        
        # AV willing users might accept slightly higher fares
        if user_profile['AV_Experience_encoded'] == 1:
            designed_reward *= 1.05
        
        cancel_prob = 0.05  # Base cancellation probability
        
        tmp = {
            'order_id': order_id,
            'origin_id': origin_id,
            'origin_lat': origin_lat,
            'origin_lng': origin_lng,
            'dest_id': dest_id,
            'dest_lat': dest_lat,
            'dest_lng': dest_lng,
            'trip_distance': trip_distance / 1000, # Convert to km
            'start_time': start_time,
            'origin_grid_id': origin_grid_id,
            'dest_grid_id': dest_grid_id,
            'itinerary_node_list': itinerary_node_list,
            'itinerary_segment_dis_list': itinerary_segment_dis_list,
            'trip_time': trip_time,
            'designed_reward': round(designed_reward, 2),
            'cancel_prob': cancel_prob,
            # Selected mode from logit model
            'selected_mode': selected_mode,
            'mode_prob_SAV': round(mode_probs['SAV'], 4),
            'mode_prob_Taxi': round(mode_probs['Taxi'], 4),
            'mode_prob_Metro': round(mode_probs['Metro'], 4),
            'mode_prob_Bus': round(mode_probs['Bus'], 4),
            # Trip attributes for selected mode
            'trip_cost': selected_attrs['cost'],
            'trip_travel_time': selected_attrs['travel_time'],
            'trip_waiting_time': selected_attrs['waiting_time'],
            # User profile categorical fields
            'user_gender': user_profile['Gender'],
            'user_age': user_profile['Age'],
            'user_income': user_profile['Income'],
            'user_education': user_profile['Education'],
            'user_household_size': user_profile['Household_Size'],
            'user_car_ownership': user_profile['Car_Ownership'],
            'user_pt_accessibility': user_profile['PT_Accessibility'],
            'user_shared_mobility_freq': user_profile['Shared_Mobility_Frequency'],
            'user_av_experience': user_profile['AV_Experience'],
            'user_sm_experience': user_profile['SM_Experience'],
            'user_sm_willingness': user_profile['SM_Willingness'],
            'user_major': user_profile['Major'],
            'user_region': user_profile['Region'],
            # User profile encoded fields (for model inference)
            'Gender_encoded': user_profile['Gender_encoded'],
            'Age_encoded': user_profile['Age_encoded'],
            'Income_encoded': user_profile['Income_encoded'],
            'Education_encoded': user_profile['Education_encoded'],
            'Household_Size_encoded': user_profile['Household_Size_encoded'],
            'Car_Ownership_encoded': user_profile['Car_Ownership_encoded'],
            'PT_Accessibility_encoded': user_profile['PT_Accessibility_encoded'],
            'Shared_Mobility_Frequency_encoded': user_profile['Shared_Mobility_Frequency_encoded'],
            'AV_Experience_encoded': user_profile['AV_Experience_encoded'],
            'SM_Experience_encoded': user_profile['SM_Experience_encoded'],
            'SM_Willingness_encoded': user_profile['SM_Willingness_encoded'],
            'Major_encoded': user_profile['Major_encoded'],
            'Region_encoded': user_profile['Region_encoded'],
            # Source scenario
            'source_scenario': scenario
        }
        
        requests.append(tmp)
        if selected_mode == 'SAV':
            filtered_requests.append(tmp)
    
    df = pd.DataFrame(requests)
    filtered_df = pd.DataFrame(filtered_requests)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    filtered_df.to_csv(output_path.replace('.csv', '_SAV_only.csv'), index=False)
    print(f"Saved {len(df)} preference-based requests to {output_path}")
    print(f"Saved {len(filtered_df)} SAV-only requests to {output_path.replace('.csv', '_SAV_only.csv')}")

    # Print summary statistics
    print("\n" + "="*60)
    print("REQUEST STATISTICS")
    print("="*60)
    
    print(f"\nTrip Statistics:")
    print(f"  Trip distance: {df['trip_distance'].min():.0f}km to {df['trip_distance'].max():.0f}km")
    print(f"  Average trip distance: {df['trip_distance'].mean():.0f}km")
    print(f"  Fare range: {df['designed_reward'].min():.2f} to {df['designed_reward'].max():.2f} HKD")
    print(f"  Average fare: {df['designed_reward'].mean():.2f} HKD")
    
    print(f"\nMode Choice Distribution:")
    mode_counts = df['selected_mode'].value_counts()
    mode_pcts = df['selected_mode'].value_counts(normalize=True)
    for mode in ['SAV', 'Taxi', 'Metro', 'Bus']:
        count = mode_counts.get(mode, 0)
        pct = mode_pcts.get(mode, 0) * 100
        print(f"  {mode}: {count} ({pct:.1f}%)")
    
    print(f"\nTarget Mode Shares (from statistics):")
    for mode, target in TARGET_MODE_SHARES.items():
        print(f"  {mode}: {target*100:.1f}%")
    
    print(f"\nDemographic Means (encoded values):")
    encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
    for col in encoded_cols:
        print(f"  avg_{col}: {df[col].mean():.2f}")
    
    print(f"\nSource scenarios:")
    print(df['source_scenario'].value_counts().to_string())
    
    return df


def export_coefficients(coefficients, output_path='./data/preference_coefficients.json'):
    """Export coefficients for use in other scripts"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coefficients, f, indent=2)
    print(f"Exported coefficients to {output_path}")


def validate_mode_shares(df, target_shares=TARGET_MODE_SHARES,
                         tol_abs=0.1, alpha=0.05, plot=True,
                         save_path='./data/mode_share_validation.png'):
    """
    Check whether the generated requests reproduce the target mode shares.

    Parameters
    ----------
    df : DataFrame with a 'selected_mode' column.
    target_shares : dict, target proportions (must sum to 1).
    tol_abs : float, max allowed absolute deviation per mode (descriptive pass/fail).
    alpha : float, significance level for the chi-square test.
    plot : bool, whether to draw a side-by-side bar chart.
    """
    modes = list(target_shares.keys())
    n = len(df)

    # Observed counts and shares (ensure every mode has an entry, even if 0)
    obs_counts = df['selected_mode'].value_counts().reindex(modes, fill_value=0)
    obs_shares = obs_counts / n
    exp_counts = pd.Series({m: target_shares[m] * n for m in modes})

    # --- Descriptive metrics ---
    abs_diff = (obs_shares - pd.Series(target_shares)).round(4)
    rel_diff = (abs_diff / pd.Series(target_shares)).round(4)
    tvd = 0.5 * rel_diff.abs().sum()

    # --- Chi-square goodness-of-fit ---
    chi2, p_value = stats.chisquare(f_obs=obs_counts.values,
                                    f_exp=exp_counts.values)

    # --- Pretty report ---
    report = pd.DataFrame({
        'observed_count'   : obs_counts.astype(int),
        'observed_share'   : obs_shares.round(4),
        'target_share'     : pd.Series(target_shares),
        'abs_diff'         : abs_diff,
        'rel_diff'         : rel_diff,
        'within_tol'       : rel_diff.abs() <= tol_abs,
    })

    print("\n" + "=" * 70)
    print("MODE SHARE VALIDATION")
    print("=" * 70)
    print(report.to_string())
    print(f"\nTotal Variation Distance : {tvd:.4f}")
    print(f"Chi-square statistic     : {chi2:.3f}  (df = {len(modes) - 1})")
    print(f"p-value                  : {p_value:.4f}")
    print(f"Decision at alpha={alpha}: "
          f"{'FAIL TO REJECT H0 (shares match target)' if p_value > alpha else 'REJECT H0 (shares differ from target)'}")
    print(f"All modes within ±{tol_abs:.0%}: {report['within_tol'].all()}")

    # --- Optional bar chart ---
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(modes))
        w = 0.38
        ax.bar(x - w/2, [target_shares[m] for m in modes], w,
               label='Target', color='#95a5a6')
        ax.bar(x + w/2, [obs_shares[m]    for m in modes], w,
               label='Observed', color='#3498db')
        for i, m in enumerate(modes):
            ax.text(i - w/2, target_shares[m] + 0.005,
                    f'{target_shares[m]:.3f}', ha='center', fontsize=9)
            ax.text(i + w/2, obs_shares[m] + 0.005,
                    f'{obs_shares[m]:.3f}', ha='center', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(modes)
        ax.set_ylabel('Share')
        ax.set_title(f'Generated vs. Target Mode Shares  '
                     f'(TVD={tvd:.3f}, p={p_value:.3f})')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved comparison plot to {save_path}")

    return {
        'report'   : report,
        'tvd'      : tvd,
        'chi2'     : chi2,
        'p_value'  : p_value,
        'passed'   : (p_value > alpha) and report['within_tol'].all(),
    }

def calibrate_ascs(coefficients,
                   target_shares=TARGET_MODE_SHARES,
                   n_sim=20000,
                   max_iter=50,
                   tol=0.005,
                   damping=1.0,
                   reference='Bus',
                   verbose=True):
    """
    ASC calibration with explicit reference correction (Train, 2009).

    For non-reference modes m:
        ASC_m += damping * [ log(target_m/pred_m) - log(target_ref/pred_ref) ]

    The reference ASC stays at 0; the correction term shifts the whole
    constant vector so that the implied reference share also matches.
    """
    coef       = dict(coefficients)
    modes      = ['SAV', 'Taxi', 'Metro', 'Bus']
    non_ref    = [m for m in modes if m != reference]
    scen_names = list(SCENARIO_WEIGHTS.keys())
    scen_probs = list(SCENARIO_WEIGHTS.values())

    history = []
    for it in range(1, max_iter + 1):
        # Monte-Carlo predicted shares
        agg = {m: 0.0 for m in modes}
        for _ in range(n_sim):
            profile  = sample_user_profile()
            scenario = random.choices(scen_names, weights=scen_probs)[0]
            _, probs = sample_mode_choice(SCENARIOS[scenario], profile, coef)
            for m in modes:
                agg[m] += probs[m]
        pred = {m: agg[m] / n_sim for m in modes}

        max_gap = max(abs(pred[m] - target_shares[m]) for m in modes)
        history.append((it, dict(pred), max_gap))
        if verbose:
            line = ", ".join(f"{m}={pred[m]:.4f}(t={target_shares[m]:.3f})"
                             for m in modes)
            print(f"[Iter {it:02d}] {line}  max|Δ|={max_gap:.4f}")

        if max_gap < tol:
            if verbose:
                print(f"Converged after {it} iterations (tol={tol}).")
            break

        # Reference correction term  (KEY FIX)
        ref_correction = math.log(
            max(target_shares[reference], 1e-6) / max(pred[reference], 1e-6)
        )

        # Update each non-reference ASC
        for m in non_ref:
            own = math.log(
                max(target_shares[m], 1e-6) / max(pred[m], 1e-6)
            )
            coef[f'ASC_{m}'] += damping * (own - ref_correction)

    if verbose:
        print("\nCalibrated ASCs:")
        for m in non_ref:
            print(f"  ASC_{m}: {coefficients[f'ASC_{m}']:+.4f}  ->  "
                  f"{coef[f'ASC_{m}']:+.4f}")

    return coef

if __name__ == '__main__':
    # 1) Calibrate ASCs to match target mode shares
    calibrated = calibrate_ascs(DEFAULT_COEFFICIENTS,
                                target_shares=TARGET_MODE_SHARES,
                                n_sim=50000,
                                max_iter=50,
                                tol=0.001,
                                damping=1)

    # 2) Generate the actual request file with the calibrated coefficients
    df = generate_preference_requests(
        num_requests=10000,
        coefficients=calibrated,
        output_path='./data/requests/preference_based_requests.csv'
    )

    export_coefficients(calibrated,
                        output_path='./data/preference_coefficients_calibrated.json')
    
    if df is not None:
        print("\nGeneration complete!")
        print(f"Coordinate ranges:")
        print(f"  Latitude: {df['origin_lat'].min():.4f} to {df['origin_lat'].max():.4f}")
        print(f"  Longitude: {df['origin_lng'].min():.4f} to {df['origin_lng'].max():.4f}")
        result = validate_mode_shares(df, TARGET_MODE_SHARES,
                                    tol_abs=0.03, alpha=0.05)
        print(f"\nValidation passed: {result['passed']}")