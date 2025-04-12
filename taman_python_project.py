import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Objective 1  --> Data Information and cleaning 
data = pd.read_csv("Public_Libraries.csv")
print(data)
print("\nDescription of the dataset: ",data.describe())
print("\nInformation of the dataset: ",data.info())
print("\nTop 5 rows of the dataset: ",data.head(5))
print("\nLast 5 rows of the dataset: ",data.tail(5))
print("\nSum of number of Duplicate values: ",data.duplicated().sum())
print("\nSum of null values: ",data.isnull().sum())
mean_rank = data["AENGLC Rank"].mean()
data.fillna({'AENGLC Rank':mean_rank},inplace=True)
mean_area = data['Population of Service Area'].mean()
data.fillna({'Population of Service Area':mean_area},inplace=True)
mean_visit = data['Total Library Visits'].mean()
data.fillna({'Total Library Visits':mean_visit},inplace=True)
mean_capita = data["Library Visits Per Capita Served"].mean()
data.fillna({"Library Visits Per Capita Served":mean_capita}, inplace = True)
mean_borrower = data["Total Registered Borrowers"].mean()
data.fillna({"Total Registered Borrowers":mean_borrower}, inplace = True)
mean_percent = data["Percent of Residents with Library Cards"].mean()
data.fillna({"Percent of Residents with Library Cards":mean_percent}, inplace = True)
mean_questions = data["Reference Questions"].mean()
data.fillna({"Reference Questions":mean_questions}, inplace = True)
mean_served = data["Reference Questions Per Capita Served"].mean()
data.fillna({"Reference Questions Per Capita Served":mean_served}, inplace = True)
mean_circul = data["Total Circulation"].mean()
data.fillna({"Total Circulation":mean_circul}, inplace = True)
mean_per = data["Circulation Per Capita Served"].mean()
data.fillna({"Circulation Per Capita Served":mean_per}, inplace = True)
mean_pre = data["Total Programs (Synchronous + Prerecorded)"].mean()
data.fillna({"Total Programs (Synchronous + Prerecorded)":mean_pre}, inplace = True)
mean_attend = data["Total Program Attendance & Views"].mean()
data.fillna({"Total Program Attendance & Views":mean_attend}, inplace = True)
mean_collection = data["Total Collection"].mean()
data.fillna({"Total Collection": mean_collection}, inplace = True)
mean_collectionp = data["Collection Per Capita Served"].mean()
data.fillna({"Collection Per Capita Served": mean_collectionp}, inplace = True)
mean_operating = data["Total Operating Income"].mean()
data.fillna({"Total Operating Income": mean_operating}, inplace = True)
mean_wages = data["Wages & Salaries Expenditures"].mean()
data.fillna({"Wages & Salaries Expenditures": mean_wages}, inplace = True)
print("\nSum of null values: ",data.isnull().sum())

#Objective 2 --> Correlation Between Numeric Features
corr = data.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()



#Objective 3 --> Distribution of Number of Libraries Across States
county_counts = data['County'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
county_counts.plot(kind='bar', color='red')
plt.title("Number of Libraries by County")
plt.xlabel("County")
plt.ylabel("Number of Libraries")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Objective 4 --> Trend of Library Establishment Over the Years
yearly_counts = data['Fiscal Year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
yearly_counts.plot(kind='line', marker='o', color='teal')
plt.title("Number of Libraries by Fiscal Year")
plt.xlabel("Fiscal Year")
plt.ylabel("Number of Libraries")
plt.grid(True)
plt.tight_layout()
plt.show()

#Objective 5 --> Outlier Detection in Total Collection
plt.figure(figsize=(5, 7))
sns.boxplot(y=data['Total Collection'], color='orange')
plt.title("Outlier Detection: Total Collection")
plt.ylabel("Total Collection")
plt.tight_layout()
plt.show()

#Objective 6 --> Distribution of Library Types
library_types = data['Principal Public?'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(library_types, labels=library_types.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
plt.title("Distribution of Library Types (Principal Public?)")
plt.tight_layout()
plt.show()

#Objective 7 --> Heatmap of Average Library Visits by County and Fiscal Year
pivot_visits = data.pivot_table(values='Total Library Visits', index='County', columns='Fiscal Year', aggfunc='mean')
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_visits, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Heatmap: Average Library Visits by County and Fiscal Year")
plt.xlabel("Fiscal Year")
plt.ylabel("County")
plt.tight_layout()
plt.show()


#Objective 8 --> Relationship Between Number of Visitors and Collection Size
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Total Collection', y='Total Library Visits', color='purple')
plt.title("Visitors vs Total Collection")
plt.xlabel("Total Collection")
plt.ylabel("Total Library Visits")
plt.tight_layout()
plt.show()

#Objective 9 --> KDE Plot of Circulation Per Capita
plt.figure(figsize=(8, 5))
sns.kdeplot(data['Circulation Per Capita Served'], fill=True, color='green')
plt.title("KDE Plot: Circulation Per Capita Served")
plt.xlabel("Circulation Per Capita Served")
plt.tight_layout()
plt.show()

#Objective 10 -->Histogram of Reference Questions
plt.figure(figsize=(8, 5))
plt.hist(data['Reference Questions'], bins=30, color='coral', edgecolor='black')
plt.title("Histogram: Reference Questions")
plt.xlabel("Reference Questions")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Objective 11 --> Line Plot of Program Attendance by Population Range
data['Population Bin'] = pd.cut(data['Population of Service Area'], bins=10)
attendance_by_pop = data.groupby('Population Bin')['Total Program Attendance & Views'].mean()
plt.figure(figsize=(10, 6))
attendance_by_pop.plot(kind='line', marker='o', color='darkblue')
plt.title("Avg. Program Attendance by Population Bin")
plt.xlabel("Population of Service Area (Binned)")
plt.ylabel("Average Program Attendance & Views")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


















