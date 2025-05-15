import folium
from folium import CustomIcon

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

    # Return the map object
    return map

def map_vessel_route(AIS_data, AIS_1_MMSI, AIS_1_SubTrackID, AIS_2_MMSI, AIS_2_SubTrackID): 
    """
    Creates an interactive map showing two AIS tracks

    Args:
        AIS_data (pd.DataFrame): AIS data 
        AIS_1_MMSI (int): MMSI of the first AIS track
        AIS_1_SubTrackID (int): SubTrackID of the first AIS track
        AIS_2_MMSI (int): MMSI of the second AIS track
        AIS_2_SubTrackID (int): SubTrackID of the second AIS track

    Returns:
        folium.Map: Interactive folium map
    """
    # Filter AIS datasets
    AIS_1 = AIS_data[(AIS_data["MMSI"] == AIS_1_MMSI) & (AIS_data["SubTrackID"] == AIS_1_SubTrackID)]
    AIS_2 = AIS_data[(AIS_data["MMSI"] == AIS_2_MMSI) & (AIS_data["SubTrackID"] == AIS_2_SubTrackID)]

    # Initialize map centered on first AIS track
    map_center = [(AIS_1["LAT"].mean() + AIS_2["LAT"].mean())/2, (AIS_1["LON"].mean() + AIS_1["LON"].mean())/2]
    map = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB Positron")

    # 1. Plot AIS 1
    AIS1_layer = folium.FeatureGroup(name=f"AIS Track 1: MMSI {AIS_1_MMSI}, SubTrackID {AIS_1_SubTrackID}")
    for _, row in AIS_1.iterrows():
        heading = 315 + row['Heading']
        icon_html = f'''
            <div style="font-size:12px; transform: rotate({heading}deg);">
                <i class="fa-solid fa-location-arrow" style="color: #ff5c5c;"></i>
            </div>
        '''
        folium.Marker(
            location=(row['LAT'], row['LON']),
            icon=folium.DivIcon(html=icon_html),
            tooltip=f"AIS Track 1<br>MMSI: {row['MMSI']}<br>Time: {row['BaseDateTime']}"
        ).add_to(AIS1_layer)
    AIS1_layer.add_to(map)

    # 2. Plot AIS 2
    AIS2_layer = folium.FeatureGroup(name=f"AIS Track 2: MMSI {AIS_2_MMSI}, SubTrackID {AIS_2_SubTrackID}")
    for _, row in AIS_2.iterrows():
        heading = 315 + row['Heading']
        icon_html = f'''
            <div style="font-size:12px; transform: rotate({heading}deg);">
                <i class="fa-solid fa-location-arrow" style="color: #117b88;"></i>
            </div>
        '''
        folium.Marker(
            location=(row['LAT'], row['LON']),
            icon=folium.DivIcon(html=icon_html),
            tooltip=f"AIS Track 2<br>MMSI: {row['MMSI']}<br>Time: {row['BaseDateTime']}"
        ).add_to(AIS2_layer)
    AIS2_layer.add_to(map)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(map)

    return map