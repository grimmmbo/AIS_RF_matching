import folium
from folium import CustomIcon
import matplotlib.colors as mcolors
import random 

def plot_coverage_area():
    """
    Creates an interactive map showing the coverage area of the AIS data
    """
    # Initialize variables
    min_longitude = -219
    max_longitude = -60
    min_latitude = 11
    max_latitude = 51.5

    coverage_area_map = folium.Map(
        location =[
            (min_latitude + max_latitude)/2, 
            (min_longitude + max_longitude)/2
        ],
        zoom_start = 2,
        tiles = "CartoDB Positron"
    )

    # Add rectangle for the bounding box
    folium.Rectangle(
        bounds = [
            [min_latitude, min_longitude], 
            [max_latitude, max_longitude]
        ],
        color = "#e14a5b",
        weight = 1,
        fill = True,
        fill_opacity = 0.2,
        popup = 'Shifted Bounding Box'
    ).add_to(coverage_area_map)

    return coverage_area_map

def plot_AIS_route(AIS_df, MMSI_to_filter_on): 
    """
    Creates an interactive map showing a vessel's route
      
    Args:
        AIS_df (pd.DataFrame): AIS data
        MMSI_to_filter_on (int): MMSI of the vessel to filter data by
    
    Returns:
        folium.Map: Folium interactive map 
    """
    # Filter datasets by target MMSI
    AIS_filtered = AIS_df[AIS_df["MMSI"] == MMSI_to_filter_on]
    
    if AIS_filtered.empty:
        raise ValueError("No data found for the specified MMSI")

    # Initialize the interactive map
    map_center = [AIS_filtered["LAT"].mean(), AIS_filtered["LON"].mean()]
    AIS_map = folium.Map(
        location = map_center, 
        zoom_start = 7, 
        tiles="CartoDB Positron"
    )

    # Plot AIS route
    for _, row in AIS_filtered.iterrows():
        heading = (315 + row["Heading"]) 
        color = "#e14a5b"
        icon_html = f'''
            <div style="font-size:12px; transform: rotate({heading}deg);">
                <i class="fa-solid fa-location-arrow" style="color: {color};"></i>
            </div>
        '''
        folium.Marker(
            location = (row['LAT'], row['LON']),
            tooltip = (
                f"AIS Point<br>MMSI: {MMSI_to_filter_on}<br>"
                f"Lat: {row['LAT']:.5f}<br>Lon: {row['LON']:.5f}<br>"
                f"On {row['BaseDateTime'].date()}, at {row['BaseDateTime'].time()}"
            ),
            icon = folium.DivIcon(html=icon_html),
        ).add_to(AIS_map)
    
    return AIS_map
    
def plot_AIS_subroutes(AIS_df, MMSI_to_filter_on): 
    """
    Creates an interactive map showing a vessel's route after it was splitted into more continuous trajectories 
      
    Args:
        AIS_df (pd.DataFrame): AIS data
        MMSI_to_filter_on (int): MMSI of the vessel to filter data by
    
    Returns:
        folium.Map: Folium interactive map 
    """
    # Filter datasets by target MMSI
    AIS_filtered = AIS_df[AIS_df["MMSI"] == MMSI_to_filter_on]
    
    if AIS_filtered.empty:
        raise ValueError("No data found for the specified MMSI")

    # Initialize the interactive map
    map_center = [AIS_filtered["LAT"].mean(), AIS_filtered["LON"].mean()]
    AIS_map = folium.Map(
        location = map_center, 
        zoom_start = 7, 
        tiles="CartoDB Positron"
    )
    
    colors = ["#e0bbe4", "#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    
    # Plot AIS route
    for i, (id, group) in enumerate(AIS_filtered.groupby(["ID"])):
        AIS_filter = folium.FeatureGroup(name=f"AIS data points for {id}")
        color = colors[i % len(colors)]
        
        for _, row in group.iterrows():
            heading = (315 + row["Heading"]) 
            icon_html = f'''
                <div style="font-size:12px; transform: rotate({heading}deg);">
                    <i class="fa-solid fa-location-arrow" style="color: {color};"></i>
                </div>
            '''
            folium.Marker(
                location = (row['LAT'], row['LON']),
                tooltip = (
                    f"AIS Point<br>MMSI: {MMSI_to_filter_on}<br>"
                    f"Lat: {row['LAT']:.5f}<br>Lon: {row['LON']:.5f}<br>"
                    f"On {row['BaseDateTime'].date()}, at {row['BaseDateTime'].time()}"
                ),
                icon = folium.DivIcon(html=icon_html),
            ).add_to(AIS_filter)
            
        AIS_filter.add_to(AIS_map)
    
    return AIS_map 

def plot_AIS_and_RF_data(AIS_data, RF_data, MMSI_to_filter):
    """
    Creates an interactive map showing a vessel's AIS and RF signals
    
    Args:
        AIS_data (pd.DataFrame): AIS data 
        RF_data (pd.DataFrame): RF data 
        MMSI_to_filter (int): MMSI of the vessel to filter data by
    
    Returns:
        folium.Map: Folium interactive map
    """
    # Make sure 'MMSI_to_filter' is a list
    if isinstance(MMSI_to_filter, int):
        MMSI_to_filter = [MMSI_to_filter]
        
    # Filter datasets by target MMSI
    AIS_filtered = AIS_data[AIS_data["ID"].isin(MMSI_to_filter)]
    RF_filtered = RF_data[(RF_data["ID"].isin(MMSI_to_filter)) & (~RF_data["RF"].isna())]
    
    # Initialize the interactive map
    map_center = [AIS_filtered["LAT"].mean(), AIS_filtered["LON"].mean()]
    map = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB Positron")

    # Kleuren per SubTrackID
    unique_ids = AIS_filtered["track_id"].unique()
    colors = ["#e14a5b", "#117b88", "#9dc6ce", "#f46d43", "#fee08b"]
    color_map = {track_id: colors[i % len(colors)] for i, track_id in enumerate(unique_ids)}

    # Plot AIS route
    ais_layer = folium.FeatureGroup(name="AIS Tracks")
    for (mmsi, track_id), group in AIS_filtered.groupby("ID"):
        color = color_map[track_id]
        for _, row in group.iterrows():
            heading = 315 + row['Heading']
            icon_html = f'''
                <div style="font-size:12px; transform: rotate({heading}deg);">
                    <i class="fa-solid fa-location-arrow" style="color: {color};"></i>
                </div>
                '''
            folium.Marker(
                location=(row['LAT'], row['LON']),
                tooltip=f"AIS Point\nMMSI: {mmsi}\nTrack: {track_id}\nTime: {row['BaseDateTime']}",
                icon=folium.DivIcon(html=icon_html)
            ).add_to(ais_layer)
    ais_layer.add_to(map)

    # Plot RF signals
    rf_layer = folium.FeatureGroup(name="RF Signals")
    for _, row in RF_filtered.iterrows():
        track_id = row["track_id"]
        color = color_map[track_id]
        icon_html = f'''
            <i class="fa-solid fa-star" style="color: {color};"></i>
            '''
        folium.Marker(
            location=row["RF"],
            tooltip=f"RF signal: {row['RF_Timestamp']}",
            icon=folium.DivIcon(html=icon_html),
        ).add_to(rf_layer)
    rf_layer.add_to(map)

    folium.LayerControl(collapsed=False).add_to(map)
    
    return map
 