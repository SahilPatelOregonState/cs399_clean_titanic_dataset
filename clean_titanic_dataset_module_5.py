#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:46:37 2025

@author: sahilvpatel1998
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import os 
from pathlib import Path

#from sklearn.preprocessing import MinMaxScaler

def print_general_info_for_df(df):
    print("Shape of the DataFrame:",df.shape)
    #print(df.columns)
    #print(df.describe())
    print("Columns:")
    col_and_types ={col:df[col].dtype for col in df.columns}
    #pprint.pprint(col_and_types.items())
    for key, value in col_and_types.items():
        print(f"{key}:[{value}]", end="\n")
    
    print("\n\n\n")
    return col_and_types

def print_unique_values_for_all_columns(df):
    unique_vals_for_cols = {col:df[col].unique() for col in  df.columns}
    for col in df.columns:
        print(f"{col}:{df[col].unique()}", end ="\n")
        
    print("\n\n\n")    
    return unique_vals_for_cols



def clean_titanic_dataset():
    #TODO:create a directory to save plots
    parent_dir = Path(__file__).resolve().parent
    save_dir = parent_dir / "saved_figures"
    # Create the directory if it does not exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    #TODO: Import raw/uncleaned csv to pandas dataframe
    filepath = "./Titanic_Kaggle_Train_Data_Dirty_OSU-2.csv"
    df = pd.read_csv(filepath)



    ##TODO: Drop Rows that I won't use
    print_general_info_for_df(df)
    df.drop(columns=["PassengerId", "Name", "Ticket" ,"Cabin"], inplace=True, axis = 1)



    #NOTE: Chose to delete passengers/rows that had bad values or missing values (only 5 in the data)
    print_general_info_for_df(df)
    print(df["Survived"].unique())
    df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce")
    Survived_column_invalid_count = (~(df["Survived"].isin([0,1]))).sum()
    print("Survived Column amount of bad values:",Survived_column_invalid_count )
    print(df["Survived"].unique())
    df= df.query("Survived == 1 or Survived == 0")
    print(df["Survived"].unique())
    print_general_info_for_df(df)


    #Plotting Fare price for spotting outliers
    #TODO CLEAN-Fare
    print("Fare column distribution(unclean)\n",df['Fare'].describe())

    #Remove the rows with huge outliers
    df["Fare"] = pd.to_numeric(df["Fare"], errors = "coerce")
    df["Fare"] = df["Fare"].astype("float")
    NaNs_in_Fare = df['Fare'].isna().sum()
    print("NaNs in Fare: ", NaNs_in_Fare)  # 0
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    fare_90th_percentile = df["Fare"].quantile(0.75)
    df["Fare"] = df["Fare"].clip(upper=fare_90th_percentile)
    df = df[df["Fare"] < 2000]
    df = df[df["Fare"] > 0]


    print("Fare column distribution(clean)\n",df['Fare'].describe())

    fare_plot = sns.boxplot(data = df, x ="Fare")
    plt.xscale("linear")
    plt.title("Distribution of Fare cleaned (Linear)")
    #plt.show(fare_plot)
    plt.savefig(f"{save_dir}/Fare_cleaned.png")

    #Normalize the data
    #scaler = MinMaxScaler()
    #df['Fare'] = scaler.fit_transform(df[['Fare']])

    #normalized_clean_fare_plot = sns.histplot(data=df, x="Fare")
    #plt.xscale("log")
    #plt.title("Distribution of Fare Cleaned  Normalized (Log)")
    #plt.show(normalized_clean_fare_plot)
    #plt.savefig("Fare_cleaned.png")

    #print(df['Fare'].describe())

    #TODO: BEFORE-Age 

    print("Age column distribution(unclean)\n",df['Age'].describe())

    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=30, kde=True, color="blue")
    plt.title("Age Distribution (uncleaned)")
    plt.xlabel("Age")
    plt.ylabel("Count")
    #plt.show()
    plt.savefig(f"{save_dir}/age_uncleaned_hist.png")

    
    #TODO: Clean Age
    #NOTE Replaced the NaN values with the mean Age and Replace the < 1 with 1
    print(df["Age"].describe())
    na_in_age = df["Age"].isna().sum()
    print("The amount of NaNs in AGE: ",na_in_age)
    invalid_age_count = df["Age"].isna().sum() + len(df.query("Age < 1"))
    print("Age Column amount of bad values:",invalid_age_count )
    median_age = int(df['Age'].median())
    df.fillna({"Age": median_age}, inplace= True)
    invalid_age_count = df["Age"].isna().sum() + len(df.query("Age < 1"))
    print("Age Column amount of bad values:",invalid_age_count )
    df.loc[df['Age'] < 1, 'Age'] = 1
    invalid_age_count = df["Age"].isna().sum() + len(df.query("Age < 1"))
    print("Age Column amount of bad values:",invalid_age_count )

    df["Age"] = df["Age"].astype("int")
    #Nobody above 130 should be on a cruise

    df = df.query("Age < 130") 
    print("The amount of NaNs in AGE: ",df["Age"].isna().sum())
    print(df["Age"].describe())

    #TODO: AFTER-AGE

    print("Age column distribution(clean)\n",df['Age'].describe())
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=30, kde=True, color="blue")
    plt.title("Age Distribution (Cleaned)")
    plt.xlabel("Age")
    plt.ylabel("Count")
    #plt.show()
    plt.savefig(f"{save_dir}/age_cleaned_hist.png")
    
    
    #TODO BEFORE-Pclass
    print("Pclass column distribution(uclean)\n",df['Pclass'].describe())
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
    plt.title("Survival by Passenger Class(Uncleaned)")
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    #plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Pclass', data=df,)
    plt.title("Survival by Passenger Class(Uncleaned)")
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    #plt.show()
    plt.savefig(f"{save_dir}/survival_by_passenger_uncleaned.png")

    #TODO: Clean PClass

    na_in_pclass = df["Pclass"].isna().sum()
    print("The amount of NaNs in Pclass: ",na_in_pclass)
    #Drop the 2 NA's
    df.dropna(subset=['Pclass'], inplace=True)
    df["Pclass"] = df["Pclass"].astype("int")
    Pclass_column_invalid_count = (~(df["Pclass"].isin([1,2,3]))).sum()
    print("Pclass Column amount of bad values:", Pclass_column_invalid_count) #0
    #df["Pclass"] = df["Pclass"].isin([1,2,3])

    #TODO: AFTER-Pclass
    print("Pclass column distribution(clean)\n",df['Pclass'].describe())
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
    plt.title("Survival by Passenger Class(Uncleaned)")
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.savefig(f"{save_dir}/survival_by_passenger_count_cleaned.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Pclass', data=df,)
    plt.title("Survival by Passenger Class(Cleaned)")
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    #plt.show()
    plt.savefig(f"{save_dir}/survival_by_passenger_box_cleaned.png")

    
    #TODO: Other Plots
    ################################################
    #TODO: Fare vs. Survival
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df['Fare'], y=df['Survived'], alpha=0.6, color="blue")
    plt.title("Scatterplot of Fare vs Survival")
    plt.xlabel("Fare")
    plt.ylabel("Survived (0 = No, 1 = Yes)")
    #plt.show()
    plt.savefig(f"{save_dir}/survival_by_passenger_scatter_cleaned.png")

    #TODO Survival Count plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=df)
    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    #plt.show()
    plt.savefig(f"{save_dir}/Survival_countplot_cleaned.png")
    ##################################################
    
    df.to_csv("Titanic_Kaggle_Train_Data_Cleaned.csv", index=False)
    print("Cleaned DataFrame saved to Titanic_Kaggle_Train_Data_Cleaned.csv")

if __name__ == "__main__":
    clean_titanic_dataset()