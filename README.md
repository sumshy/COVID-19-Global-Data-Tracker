# 🦠 COVID-19 Global Data Tracker

## 📌 Project Title:
**COVID-19 Global Data Tracker using Python**

## 📝 Project Description

This project analyzes global COVID-19 trends using real-world data from [Our World in Data](https://ourworldindata.org/coronavirus). 
The goal is to track total cases, deaths, and vaccinations across time and countries — specifically **Kenya**, **United States**, and **India** — 
by performing data cleaning, analysis, and creating clear visualizations using Python.
By the end of this project, we produce a full data report with statistical summaries, graphs, and insights that tell the story behind the numbers. 
This is built using a standalone Python script (`finalproject.py`) and can easily be extended or adapted.

## 🎯 Project Objectives

- ✅ Import and clean COVID-19 global data
- ✅ Analyze time trends (cases, deaths, vaccinations)
- ✅ Compare metrics across countries
- ✅ Visualize trends using Matplotlib and Seaborn
- ✅ Communicate findings with statistical and visual storytelling

## 📂 Dataset

- **Source**: [Our World in Data - COVID-19](https://ourworldindata.org/coronavirus)
- **File Name**: `owid-covid-data.csv`
- **Download Link**: [Download CSV Directly](https://covid.ourworldindata.org/data/owid-covid-data.csv)

Save the file in the same folder as `finalproject.py` before running the script.

## ⚙️ Requirements

Make sure to have Python 3.x and install the following libraries:
pip install pandas matplotlib seaborn

**🚀 How to Run**
Download the dataset and save it as owid-covid-data.csv.

Open the finalproject.py script in VS Code.

Run the script.

The terminal will display summaries and insights, while charts will open in separate windows.

📈 Visualizations Included
Visualization Type	Description
📉 Line Chart	Tracks total COVID-19 cases over time by country
📊 Bar Chart	Compares average total deaths by country
📊 Histogram	Shows distribution of total vaccinations
🔘 Scatter Plot	Shows correlation between total cases and total deaths

**🧠 Findings & Observations**
After analyzing the COVID-19 data for Kenya, the United States, and India, here are the key insights:

**Case Trends Over Time:**

The U.S. showed a steep and steady rise in cases.
India had distinct surges, particularly in 2021.
Kenya’s curve remained comparatively low.

**Deaths Comparison:**

The U.S. had the highest average deaths, followed by India.
Kenya showed a significantly lower mortality count.

**Vaccination Distribution:**

The U.S. led in total vaccinations, with India following.
Kenya lagged far behind due to vaccine access and logistics.

**Death Rate Insights:**

The death rate stayed low overall.
The U.S. displayed a higher case-fatality ratio compared to the other countries.

**Correlation Observation:**

A clear positive correlation between total cases and deaths confirmed the burden scale in high-case countries.

**🧰 Tools Used**
pandas for data handling and analysis
matplotlib and seaborn for data visualization
try-except error handling for reliable script execution
Clean code with modular structure and inline explanations

**📌 Deliverables**
✅ Python Script: finalproject.py

✅ Dataset: owid-covid-data.csv

✅ Visualizations (automatically generated)

✅ Insights included in this README for reporting and presentation


