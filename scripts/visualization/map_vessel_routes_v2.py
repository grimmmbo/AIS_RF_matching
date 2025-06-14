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
        color = "#ff5c5c",
        weight = 1,
        fill = True,
        fill_opacity = 0.2,
        popup = 'Shifted Bounding Box'
    ).add_to(coverage_area_map)

    return coverage_area_map

def plot_AIS_route(AIS_df, MMSI_to_filter_on): 
    """
    Creates an interactive map showing comparison of a vessel's routes based on mutliple datasets
      
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
        color = "#ff5c5c"
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
    

    