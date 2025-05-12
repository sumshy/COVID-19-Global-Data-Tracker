import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and explore the dataset
try:
    df = pd.read_csv("owid-covid-data.csv")

    print("Dataset loaded successfully.\n")
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\n Column names:")
    print(df.columns)

    print("\n🔍 Dataset information:")
    print(df.info())

    print("\n Missing values in each column:")
    print(df.isnull().sum())
except FileNotFoundError:
    print("The file 'owid-covid-data.csv' was not found. Please ensure it's in the same folder.")
except Exception as e:
    print(f"An error occurred: {e}")


#Cleaning and preparing the data


df['date'] = pd.to_datetime(df['date'])

# Filter the dataset for selected countries
countries = ['Kenya', 'United States', 'India']
df = df[df['location'].isin(countries)]

# Drop rows where any of the critical columns are missing
critical_columns = ['total_cases', 'total_deaths', 'total_vaccinations']
df = df.dropna(subset=critical_columns)

# Fill remaining missing values using forward fill method
df.fillna(method='ffill', inplace=True)

# Confirm cleaning results
print("\nData cleaned and filtered.")
print(f"Countries in focus: {countries}")
print(f"Dataset now has {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


# Basic Data Analysis

# Show descriptive statistics for numerical columns
print("\nBasic Statistical Summary:")
print(df.describe())

# Group by country and calculate average of key metrics
grouped_stats = df.groupby('location')[['total_cases', 'total_deaths', 'total_vaccinations']].mean()

print("\nAverage values per country:")
print(grouped_stats)

# Calculate death rate as a new column
df['death_rate'] = df['total_deaths'] / df['total_cases']

# Check if the death rate was added successfully
print("\nDeath rate column has been added.")
print(df[['location', 'date', 'total_cases', 'total_deaths', 'death_rate']].tail())


# Set plot style
sns.set(style="whitegrid")

# Line Chart: Total COVID-19 cases over time by country
plt.figure(figsize=(12, 6))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart: Average total deaths per country
avg_deaths = df.groupby('location')['total_deaths'].mean()
plt.figure(figsize=(8, 6))
sns.barplot(x=avg_deaths.index, y=avg_deaths.values)
plt.title('Average Total Deaths per Country')
plt.xlabel('Country')
plt.ylabel('Average Total Deaths')
plt.tight_layout()
plt.show()

# Histogram: Distribution of total vaccinations
plt.figure(figsize=(8, 6))
sns.histplot(df['total_vaccinations'], bins=20, kde=True)
plt.title('Distribution of Total Vaccinations')
plt.xlabel('Total Vaccinations')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plot: Total Cases vs Total Deaths
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='total_cases', y='total_deaths', hue='location')
plt.title('Total Cases vs Total Deaths')
plt.xlabel('Total Cases')
plt.ylabel('Total Deaths')
plt.legend(title='Country')
plt.tight_layout()
plt.show()

print("\nAll visualizations generated successfully.")
