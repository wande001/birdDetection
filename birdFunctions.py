import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

exclude = ["Dog", "Human non-vocal", "Engine", "Human vocal",
           "Trompetzwaan", "Viskraai", "Waaierhoen", "Prairiehoen",
           "Elzenfeetiran", "Amerikaanse Nachtzwaluw", "Cederpestvogel",
            "Casarca", "Cassins Vireo", "Blonde Ruiter","Kleine Torenvalk",
            "Gevlekte Diamantvogel","Geelstuitdoornsnavel","Gestreepte Bosuil",
            "Hoatzin","Kleine Kauailijster","Oeraluil","Roodkapzanger","Pinyongaai",
            "Rosse Bladspeurder","Oehoe","Gray Wolf","Amerikaanse Oehoe",
            "Ponderosadwergooruil","Rotsduif", "Porseleinhoen", "Roerdomp",
            "Amerikaanse Woudaap", "Geelkopamazone", "Zwarte Dwergral", "Southern Boobook",
            "Roodkopspecht", "Lesson's Motmot","Chinese Spoorkoekoek","Berkenfeetiran",
            "Canadese Boomklever", "Avonddikbek", "Japanse Koolmees", "Carolina-eend",
            "Haakbek", "Grijze Mees", "Fireworks", "Verreaux' Duif", "Europese Kanarie"]

def readCSV():
    df = pd.read_csv("data/output.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def filterData(minLim=60, confidence_threshold=0.75):
    # --- Filter last hour ---
    df = readCSV()
    endTime = (df['timestamp'].iloc[-1] - pd.Timedelta(minutes=2)).ceil("60min")
    timeFrame = endTime - timedelta(minutes=minLim)
    df_selection = df[df['timestamp'] >= timeFrame]
    # --- Filter high confidence ---
    df_selection = df_selection[df_selection['confidence'] > confidence_threshold]
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

def makeHeatmap(data, timeFrame="5min"):
    # Convert timestamps to minute resolution
    # Determine the last hour time range
    if timeFrame == "5min":
        data["minute"] = data["timestamp"].dt.floor(timeFrame)
        end_time = (data['timestamp'].iloc[-1] - pd.Timedelta(minutes=5)).ceil("60min")
        start_time = end_time - pd.Timedelta(minutes=65) + pd.Timedelta(minutes=5)
        all_blocks = pd.date_range(start=start_time, end=end_time, freq=timeFrame)
    elif timeFrame == "hour":
        data["minute"] = data["timestamp"].dt.floor("60min")
        end_time = (data['timestamp'].iloc[-1] - pd.Timedelta(minutes=5)).ceil("60min")
        start_time = end_time - pd.Timedelta(minutes=60*24) + pd.Timedelta(minutes=60)
        all_blocks = pd.date_range(start=start_time, end=end_time, freq="60min")
    elif timeFrame == "day":
        data["minute"] = data["timestamp"].dt.floor("1440min")
        end_time = (data['timestamp'].iloc[-1] - pd.Timedelta(minutes=5)).floor("1440min")
        start_time = end_time - pd.Timedelta(minutes=60*24*7) + pd.Timedelta(minutes=60*24)
        all_blocks = pd.date_range(start=start_time, end=end_time, freq="1440min")
    elif timeFrame == "week":
        data["minute"] = data["timestamp"].dt.floor("1440min")
        end_time = (data['timestamp'].iloc[-1] - pd.Timedelta(minutes=5)).floor("1440min")
        start_time = data["timestamp"].dt.floor("1440min").min()
        all_blocks = pd.date_range(start=start_time, end=end_time, freq="1440min")


    # Create pivot: species (rows) × minute (columns)
    heatmap_data = (
        data
        .groupby(["species", "minute"])
        .size()
        .reset_index(name="count")
        .pivot(index="species", columns="minute", values="count")
        .reindex(columns=all_blocks)   # ensures 12 blocks
        .fillna(np.nan)
    )
    heatmap_data.loc["Total"] = heatmap_data.sum(axis=0)
    heatmap_data = heatmap_data.loc[
        heatmap_data.sum(axis=1).sort_values(ascending=True).index
    ]
    heatmap_normalized = heatmap_data.div(heatmap_data.max(axis=1), axis=0)

    # Convert to matrix form for matplotlib
    heat_matrix = heatmap_normalized.values

    # Create figure
    if timeFrame == "5min": plt.figure(figsize=(5, 6))
    if timeFrame == "hour": plt.figure(figsize=(5, 14))
    if timeFrame == "day": plt.figure(figsize=(5, 14))
    if timeFrame == "week": plt.figure(figsize=(5, 25))

    # ---- PLOT WITH MATPLOTLIB ONLY ----
    plt.imshow(heat_matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues", origin="lower")

    # Add colorbar
    #plt.colorbar(label="Observation Count")

    # Tick labels
    if timeFrame == "5min": plotLabels = [m.strftime("%H:%M") for m in heatmap_data.columns]
    elif timeFrame == "hour": plotLabels = [m.strftime("%H:%M") for m in heatmap_data.columns]
    elif timeFrame == "day": plotLabels = [m.strftime("%D") for m in heatmap_data.columns]
    elif timeFrame == "week": plotLabels = [m.strftime("%D") for m in heatmap_data.columns]
    plt.xticks(
        ticks=range(len(heatmap_data.columns)),
        labels=plotLabels,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        ticks=range(len(heatmap_data.index)),
        labels=["%s %d" % (species, total) for species, total in zip(heatmap_data.index, heatmap_data.sum(axis=1))]
    )
    plt.title("Species Presence Heatmap")
    plt.tight_layout()
    plt.show()

def shannon_diversity(data):
    data['hour_of_day'] = data['timestamp'].dt.hour

    hour_groups = data.groupby('hour_of_day')['species'].apply(list)
    diversity = hour_groups.apply(diversityLoop)

    # Plot
    plt.figure(figsize=(5,5))
    diversity.plot()
    plt.xlabel('Hour')
    plt.ylabel('Acoustic Diversity')
    plt.title('')
    plt.show()

def diversityLoop(species_list):
    counts = pd.Series(species_list).value_counts()
    freqs = counts / counts.sum()
    return -np.sum(freqs * np.log2(freqs))

def makeDailyCycleHeatmap(data):
    # Convert timestamp to hour of day
    data['hour_of_day'] = data['timestamp'].dt.hour

    # Count calls per species per hour
    heatmap_data = data.groupby(['species', 'hour_of_day']).size().unstack(fill_value=0)

    # Order species by total occurrence (descending)
    species_order = heatmap_data.sum(axis=1).sort_values(ascending=True).index
    heatmap_data = heatmap_data.reindex(species_order)

    # Normalize per species for better visualization
    heatmap_normalized = heatmap_data.div(heatmap_data.max(axis=1), axis=0)

    # Replace zeros with np.nan for better visualization
    heatmap_normalized = heatmap_normalized.replace(0, np.nan)

    # Convert to matrix
    heat_matrix = heatmap_normalized.values

    # Plot
    plt.figure(figsize=(5,25))
    plt.imshow(heat_matrix, aspect='auto', cmap='Blues', origin='lower')

    # X-axis: hours
    x_labels = [f"{int(t):02d}:00" for t in heatmap_normalized.columns]
    plt.xticks(ticks=range(len(heatmap_normalized.columns)), labels=x_labels, rotation=45, ha='right')

    # Y-axis: species names + total counts
    y_labels = [f"{species} ({int(total)})" for species, total in zip(heatmap_normalized.index, heatmap_data.sum(axis=1))]
    plt.yticks(ticks=range(len(heatmap_normalized.index)), labels=y_labels)

    plt.title("Daily Cycle Heatmap")
    plt.xlabel("Hour of Day")
    plt.tight_layout()
    plt.show()