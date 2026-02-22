import pandas as pd
from datetime import datetime, timedelta

exclude = ["Dog", "Human non-vocal", "Engine", "Human vocal",
           "Trompetzwaan", "Viskraai", "Waaierhoen", "Prairiehoen",
           "Elzenfeetiran", "Amerikaanse Nachtzwaluw", "Cederpestvogel",
            "Casarca", "Cassins Vireo", "Blonde Ruiter","Kleine Torenvalk",
            "Gevlekte Diamantvogel","Geelstuitdoornsnavel","Gestreepte Bosuil",
            "Hoatzin","Kleine Kauailijster","Oeraluil","Roodkapzanger","Pinyongaai",
            "Rosse Bladspeurder","Oehoe","Gray Wolf","Amerikaanse Oehoe",
            "Ponderosadwergooruil","Rotsduif"]

def readCSV():
    df = pd.read_csv("data/output.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def filterData(minLim=60, confidence_threshold=0.75):
    # --- Filter last hour ---
    df = readCSV()
    timeFrame = datetime.now() - timedelta(minutes=minLim)
    df_selection = df[df['timestamp'] >= timeFrame]
    # --- Filter high confidence ---
    df_selection = df_selection[df_selection['confidence'] > confidence_threshold]
    exclude = ["Dog", "Human non-vocal", "Engine", "Human-vocal"]
    df_selection = df_selection[~df_selection['species'].isin(exclude)]
    return df_selection

def aggregateData(df_selection):
    # --- Aggregate by species ---
    agg_data = df_selection.groupby("species").size().reset_index(name="count").sort_values("count", ascending=False).reset_index(drop=True) 
    return agg_data

def groupByMinute(df):
    # --- Group by minute ---
    stats = df.groupby(df['timestamp'].dt.minute).size()
    stats = (stats.reindex(range(60), fill_value=0))
    return stats

def groupByHour(df):
    # --- Group by hour ---
    stats = df.groupby(df['timestamp'].dt.hour).size()
    stats = (stats.reindex(range(24), fill_value=0))
    return stats

def groupByDay(df):
    # --- Group by day ---
    stats = df.groupby(df['timestamp'].dt.date).size()
    return stats