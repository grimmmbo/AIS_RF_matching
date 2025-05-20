import folium
from folium import CustomIcon
import matplotlib.colors as mcolors
import random 

def compare_vessel_tracks_on_map(normal_AIS_data, modified_AIS_data, RF_data, MMSI_to_filter): 
    """
    Creates an interactive map showing comparison of a vessel's routes based on mutliple datasets
      
    Args:
        normal_AIS_data (pd.DataFrame): Normal AIS data
        modified_AIS_data (pd.DataFrame): Modified AIS data
        RF_data (pd.DataFrame): RF data 
        MMSI_to_filter (int): MMSI of the vessel to filter data by
    
    Returns:
        folium.Map: Folium interactive map 
    """
    # Filter datasets by target MMSI
    normal_AIS_data_filtered = normal_AIS_data[normal_AIS_data["MMSI"] == MMSI_to_filter]
    modified_AIS_data_filtered = modified_AIS_data[modified_AIS_data["MMSI"] == MMSI_to_filter]
    RF_data_filtered = RF_data[RF_data["MMSI"] == MMSI_to_filter]

    # Initialize the interactive map
    latitude_of_center = (
        normal_AIS_data_filtered["LAT"].mean() + 
        modified_AIS_data_filtered["LAT"].mean() + 
        RF_data_filtered["RF_LAT"].mean()
    ) / 3
    
    longitude_of_center = (
        normal_AIS_data_filtered["LON"].mean() + 
        modified_AIS_data_filtered["LON"].mean() + 
        RF_data_filtered["RF_LON"].mean()
    ) / 3
    
    map = folium.Map(location=[latitude_of_center, longitude_of_center], zoom_start=7, tiles="CartoDB Positron")
    
    # Initialize the colors of the interactive map
    colors = {
        0: {"color": "#ff5c5c"},
        1: {"color": "#c31e24"},
        2: {"color": "#9dc6ce"},
        3: {"color": "#117b88"}}

    # 1. Plot 'normal' AIS data points 
    normal_AIS_filter = folium.FeatureGroup(name="Normal AIS Tracks")
    
    for (mmsi, track_id), group in normal_AIS_data_filtered.groupby(['MMSI', 'SubTrackID']):
        color = colors[track_id]["color"]
        
        for _, row in group.iterrows():
            heading = 315 + row['Heading']
            icon_html = f'''
                <div style="font-size:12px; transform: rotate({heading}deg);">
                    <i class="fa-solid fa-location-arrow" style="color: {color};"></i>
                </div>
                '''
            folium.Marker(
                location=(row['LAT'], row['LON']),
                tooltip=f"Normal AIS data point<br>Belongs to vessel with MMSI: {mmsi}<br>Lat: {row['LAT']}<br>Lon: {row['LON']}<br>Subtrack ID: {track_id}<br>Timestamp: {row['BaseDateTime']}",
                icon=folium.DivIcon(html=icon_html)
            ).add_to(normal_AIS_filter)
            
    normal_AIS_filter.add_to(map)

    # 2. Plot 'modified' AIS data points
    modified_AIS_filter = folium.FeatureGroup(name="Manipulated AIS Tracks")
    
    for (mmsi, track_id), group in modified_AIS_data_filtered.groupby(['MMSI', 'SubTrackID']):
        color = colors[track_id]["color"]
        
        for _, row in group.iterrows():
            heading = 315 + row['Heading']
            icon_html = f'''
                <div style="font-size:12px; transform: rotate({heading}deg);">
                    <i class="fa-solid fa-location-arrow" style="color: {color};"></i>
                </div>
                '''
            folium.Marker(
                location=(row['LAT'], row['LON']),
                icon=folium.DivIcon(html=icon_html),
                tooltip=f"Modified AIS data point<br>Belongs to vessel with MMSI: {mmsi}<br>Lat: {row['LAT']}<br>Lon: {row['LON']}<br>Subtrack ID: {track_id}<br>Timestamp: {row['BaseDateTime']}"
            ).add_to(modified_AIS_filter)
            
    modified_AIS_filter.add_to(map)

    # 3. Simulated RF Signals
    RF_filter = folium.FeatureGroup(name="Simulated RF Signals")
    for (mmsi, timestamp), group in RF_data_filtered.groupby(['MMSI', 'TimeStamp_RF']):
        
        for _, row in group.iterrows():
            custom_icon = CustomIcon(
                icon_image="../images/star_image_yellow.png",
                icon_size=(30, 30),
                icon_anchor=(15, 15)
            )
            folium.Marker(
                location=[row['RF_LAT'], row['RF_LON']],
                icon=custom_icon,
                tooltip=f"Simulated RF signal at {row['TimeStamp_RF']}<br>Belongs to the following AIS data:<br>Lat: {row['AIS_LAT']}<br>Lon: {row['AIS_LON']}<br>Subtrack ID: {row['SubTrackID']}"
            ).add_to(RF_filter)
            
    RF_filter.add_to(map)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(map)

    return map

def extract_plot_data_by_MMSI(AIS_data, MMSI_list, RF_data = None, include_AIS = True, include_RF = False):
    """
    Extracts AIS and/or RF data for a given list of MMSI's
    
    Args:
        AIS_data (pd.DataFrame): AIS data
        RF_data (pd.DataFrame): RF data, defaults to None
        MMSI_list (list): List of MMSI numbers to filter by
        include_AIS (bool): Whether to include AIS data points in the plot, defaults to True
        include_RF (bool): Whether to include RF data points in the plot, defaults to False

    Returns:
        tuple: 
            pd.DataFrame: Filtered AIS data
            pd.DataFrame: Filtered RF data
    """
    filtered_AIS = None
    filtered_RF = None
    
    if include_AIS and AIS_data is not None:
        filtered_AIS = AIS_data[AIS_data.apply(lambda row: (row["MMSI"], row["SubTrackID"]) in MMSI_list, axis=1)]
    
    if include_RF and RF_data is not None:
        filtered_RF = RF_data[RF_data.apply(lambda row: (row["MMSI"], row["SubTrackID"]) in MMSI_list, axis=1)]
    
    return filtered_AIS, filtered_RF 

def map_vessel_route(AIS_data, MMSI_list, RF_data = None, include_AIS = True, include_RF = False): 
    """
    Plots the routes of vessels on an interactive map using Folium

    Args:
        AIS_data (pd.DataFrame): AIS data
        RF_data (pd.DataFrame): RF data, defaults to None
        MMSI_list (list): List of MMSI numbers to filter by
        include_AIS (bool): Whether to include AIS data points in the plot, defaults to True
        include_RF (bool): Whether to include RF data points in the plot, defaults to False

    Returns:
        folium.Map: Interactive folium map
    """
    # Get the data points for the specified MMSI
    AIS_data, RF_data = extract_plot_data_by_MMSI(AIS_data, MMSI_list, RF_data, include_AIS, include_RF)
    
    # Initialize map
    if AIS_data is not None:
        map_center = [AIS_data["LAT"].mean(), AIS_data["LON"].mean()]
    elif RF_data is not None:
        map_center = [RF_data["RF_LAT"].mean(), RF_data["RF_LON"].mean()]
    elif AIS_data is not None and RF_data is not None:
        map_center = [(AIS_data["LAT"].mean() + RF_data["RF_LAT"].mean())/2, (AIS_data["LON"].mean() + RF_data["RF_LON"].mean())/2]
            
    map = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB Positron")
    
    # Initialize the colors of the interactive map
    colors = list(mcolors.CSS4_COLORS.keys()) 

    # 1. Plot AIS data points 
    if AIS_data is not None:
        
        for i, ((mmsi, track_id), group) in enumerate(AIS_data.groupby(['MMSI', 'SubTrackID'])):
            AIS_filter = folium.FeatureGroup(name=f"AIS data points {mmsi}")
            color = random.choice(colors)
            
            for _, row in group.iterrows():
                heading = 315 + row['Heading']
                icon_html = f'''
                    <div style="font-size:12px; transform: rotate({heading}deg);">
                        <i class="fa-solid fa-location-arrow" style="color: {color};"></i>
                    </div>
                    '''
                folium.Marker(
                    location=(row['LAT'], row['LON']),
                    tooltip=f"AIS data point belongs to vessel with: <br> MMSI: {mmsi}<br> Timestamp: {row['BaseDateTime']}",
                    icon=folium.DivIcon(html=icon_html)
                ).add_to(AIS_filter)
                
            AIS_filter.add_to(map)

    # 2. Plot RF data points 
    if RF_data is not None:
        RF_filter = folium.FeatureGroup(name="RF data points")
        
        for (mmsi, timestamp), group in RF_data.groupby(['MMSI', 'TimeStamp_RF']):
            for _, row in group.iterrows():
                custom_icon = CustomIcon(
                    icon_image="../images/star_image_yellow.png",
                    icon_size=(30, 30),
                    icon_anchor=(15, 15)
                )
                folium.Marker(
                    location=[row['RF_LAT'], row['RF_LON']],
                    icon=custom_icon,
                    tooltip=f"RF data point belongs to vessel with: <br> MMSI: {mmsi}<br> Timestamp: {row['TimeStamp_RF']}",
                ).add_to(RF_filter)
                
        RF_filter.add_to(map)
        
    # Layer control
    folium.LayerControl(collapsed=False).add_to(map)

    return map