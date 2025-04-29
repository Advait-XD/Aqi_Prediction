import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers(data, method='iqr', threshold=1.5):
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return data[(z_scores < threshold).all(axis=1)]

def calculate_aqi_single(concentration, breakpoints):
    for (bp_low, bp_high), (i_low, i_high) in breakpoints:
        if bp_low <= concentration <= bp_high:
            return ((i_high - i_low) / (bp_high - bp_low)) * (concentration - bp_low) + i_low
    return None

def calculate_aqi(data, outlier_method='iqr', outlier_threshold=1.5):
    data = remove_outliers(data.select_dtypes(include=[np.number]), method=outlier_method, threshold=outlier_threshold)

    breakpoints = {
        'PM2.5': [
            ((0, 12.0), (0, 50)),
            ((12.1, 35.4), (51, 100)),
            ((35.5, 55.4), (101, 150)),
            ((55.5, 150.4), (151, 200)),
            ((150.5, 250.4), (201, 300)),
            ((250.5, 500.4), (301, 500))
        ],
        'PM10': [
            ((0, 54), (0, 50)),
            ((55, 154), (51, 100)),
            ((155, 254), (101, 150)),
            ((255, 354), (151, 200)),
            ((355, 424), (201, 300)),
            ((425, 604), (301, 500))
        ],
        'NO': [
            ((0, 0.053), (0, 50)),
            ((0.054, 0.1), (51, 100)),
            ((0.1, 0.2), (101, 150)),
            ((0.2, 0.5), (151, 200)),
            ((0.5, 1.0), (201, 300)),
            ((1.0, 2.0), (301, 500))
        ],
        'NO2': [
            ((0, 53), (0, 50)),
            ((54, 100), (51, 100)),
            ((101, 360), (101, 150)),
            ((361, 649), (151, 200)),
            ((650, 1249), (201, 300)),
            ((1250, 2049), (301, 500))
        ],
        'NOx': [
            ((0, 0.1), (0, 50)),
            ((0.1, 0.2), (51, 100)),
            ((0.2, 0.5), (101, 150)),
            ((0.5, 1.0), (151, 200)),
            ((1.0, 2.0), (201, 300)),
            ((2.0, 5.0), (301, 500))
        ],
        'NH3': [
            ((0, 0.2), (0, 50)),
            ((0.21, 0.5), (51, 100)),
            ((0.51, 1.0), (101, 150)),
            ((1.01, 2.0), (151, 200)),
            ((2.01, 5.0), (201, 300)),
            ((5.01, 10.0), (301, 500))
        ],
        'CO': [
            ((0, 4.4), (0, 50)),
            ((4.5, 9.4), (51, 100)),
            ((9.5, 12.4), (101, 150)),
            ((12.5, 15.4), (151, 200)),
            ((15.5, 30.4), (201, 300)),
            ((30.5, 50.0), (301, 500))
        ],
        'SO2': [
            ((0, 35), (0, 50)),
            ((36, 75), (51, 100)),
            ((76, 185), (101, 150)),
            ((186, 304), (151, 200)),
            ((305, 604), (201, 300)),
            ((605, 1000), (301, 500))
        ],
        'O3': [
            ((0, 54), (0, 50)),
            ((55, 70), (51, 100)),
            ((71, 85), (101, 150)),
            ((86, 105), (151, 200)),
            ((106, 200), (201, 300)),
            ((201, 300), (301, 500))
        ],
        'Benzene': [
            ((0, 0.1), (0, 50)),
            ((0.11, 0.5), (51, 100)),
            ((0.51, 1.0), (101, 150)),
            ((1.0, 2.0), (151, 200)),
            ((2.1, 5.0), (201, 300)),
            ((5.1, 10.0), (301, 500))
        ],
        'Toluene': [
            ((0, 0.1), (0, 50)),
            ((0.11, 0.5), (51, 100)),
            ((0.51, 1.0), (101, 150)),
            ((1.01, 2.0), (151, 200)),
            ((2.01, 5.0), (201, 300)),
            ((5.01, 10.0), (301, 500))
        ],
        'Xylene': [
            ((0, 0.1), (0, 50)),
            ((0.11, 0.5), (51, 100)),
            ((0.51, 1.0), (101, 150)),
            ((1.01, 2.0), (151, 200)),
            ((2.01, 5.0), (201, 300)),
            ((5.01, 10.0), (301, 500))
        ]
    }

    aqi_results = {}
    for pollutant, concentrations in breakpoints.items():
        if pollutant in data.columns:
            aqi = calculate_aqi_single(data[pollutant].mean(), concentrations)
            if aqi is not None:  # Only store valid AQI values
                aqi_results[pollutant] = aqi

    if aqi_results:
        overall_aqi = max(aqi_results.values())
    else:
        overall_aqi = None

    return overall_aqi

def main():
    data = pd.DataFrame({
        'PM2.5': [10, 20, 30, 40, 50],
        'PM10': [20, 30, 40, 50, 60],
        'NO': [0.01, 0.02, 0.03, 0.04, 0.05],
        'NO2': [10, 20, 30, 40, 50],
        'NOx': [0.1, 0.2, 0.3, 0.4, 0.5],
        'NH3': [0.1, 0.2, 0.3, 0.4, 0.5],
        'CO': [1, 2, 3, 4, 5],
        'SO2': [10, 20, 30, 40, 50],
        'O3': [50, 60, 70, 80, 90],
        'Benzene': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Toluene': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Xylene': [0.1, 0.2, 0.3, 0.4, 0.5]
    })

    overall_aqi = calculate_aqi(data)
    print(f"Overall AQI: {overall_aqi}")

if __name__ == "__main__":
    main()