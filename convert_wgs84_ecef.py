import pandas as pd
import numpy as np

# Define constants for WGS84
a = 6378137.0           # Semi-major axis of the Earth in meters
f = 1 / 298.257223563   # Flattening
e2 = f * (2 - f)        # Square of eccentricity

def geodetic_to_ecef(lat, lon, alt=0):
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF coordinates (X, Y, Z).
    
    Parameters:
    lat (float): Latitude in degrees
    lon (float): Longitude in degrees
    alt (float): Altitude in meters

    Returns:
    tuple: ECEF coordinates (X, Y, Z)
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Calculate the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    
    return X, Y, Z

# Load the CSV data
file_path = 'anomaly_circle.csv'  # Update this to your actual file path
df = pd.read_csv(file_path)

# Calculate ECEF coordinates for each row in the DataFrame
ecef_coords = df.apply(lambda row: geodetic_to_ecef(row['latitude'], row['longitude'], row.get('altitude', 0)), axis=1)

# Add X, Y, and Z columns to the DataFrame
df[['X', 'Y', 'Z']] = pd.DataFrame(ecef_coords.tolist(), index=df.index)

# Save the updated DataFrame back to a CSV file
output_file_path = 'updated_anomaly_circle_with_ecef.csv'
df.to_csv(output_file_path, index=False)

print(f"Updated data saved to {output_file_path}")
