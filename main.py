import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

if __name__ == '__main__':
    raw_data = pd.read_csv("GlobalTemperatures.csv", parse_dates=['dt'])
    date_and_avg_temp = raw_data[['dt', 'LandAverageTemperature']]
    print(date_and_avg_temp)

    plt.scatter(date_and_avg_temp['dt'], date_and_avg_temp['LandAverageTemperature'], color='red')
    plt.title("Temp by date")
    plt.show()

    avg_temp_by_year = date_and_avg_temp.groupby(date_and_avg_temp['dt'].dt.year)[
        'LandAverageTemperature'].mean().reset_index()
    print(avg_temp_by_year)

    plt.scatter(avg_temp_by_year['dt'], avg_temp_by_year['LandAverageTemperature'], color='blue')
    plt.title("Temp by year")
    plt.show()

    regression = linear_model.LinearRegression()

    regression.fit(avg_temp_by_year['dt'].values.reshape(-1, 1), avg_temp_by_year['LandAverageTemperature'])
    # prediction = regression.predict([[2023], [2024], [2025], [2026], [2027], [2028]])
    prediction = regression.predict(avg_temp_by_year['dt'].values.reshape(-1, 1))
    plt.scatter(avg_temp_by_year['dt'], avg_temp_by_year['LandAverageTemperature'], color='blue')
    plt.plot(avg_temp_by_year['dt'], prediction, color='red')
    plt.title("Prediction vs data")
    plt.show()
