import folium
import pandas as pd

# Load cleaned SpaceX dataset
df = pd.read_csv("../data/spacex_cleaned_data.csv")

# Create base map
space_x_map = folium.Map(
    location=[28.5, -80.6],
    zoom_start=5
)

# Example launch-site information
launch_sites = [
    {
        "name": "CCAFS SLC-40",
        "latitude": 28.5618571,
        "longitude": -80.577366,
        "launches": 50
    },
    {
        "name": "KSC LC-39A",
        "latitude": 28.6080585,
        "longitude": -80.604046,
        "launches": 30
    },
    {
        "name": "VAFB SLC-4E",
        "latitude": 34.632093,
        "longitude": -120.610829,
        "launches": 20
    }
]

# Add markers and circle markers
for site in launch_sites:

    popup_text = f"""
    <b>Launch Site:</b> {site['name']}<br>
    <b>Number of Launches:</b> {site['launches']}
    """

    # Standard marker
    folium.Marker(
        location=[
            site["latitude"],
            site["longitude"]
        ],
        popup=folium.Popup(
            popup_text,
            max_width=300
        ),
        tooltip=site["name"]
    ).add_to(space_x_map)

    # Circle marker
    folium.CircleMarker(
        location=[
            site["latitude"],
            site["longitude"]
        ],
        radius=max(5, site["launches"] / 3),
        popup=popup_text,
        tooltip=f"{site['name']} - {site['launches']} launches",
        fill=True
    ).add_to(space_x_map)

# Add a line connecting launch sites
coordinates = [
    [site["latitude"], site["longitude"]]
    for site in launch_sites
]

folium.PolyLine(
    locations=coordinates,
    weight=3,
    tooltip="SpaceX Launch Site Locations"
).add_to(space_x_map)

# Add layer control
folium.LayerControl().add_to(space_x_map)

# Display map
space_x_map

# Save the interactive map as an HTML file
space_x_map.save(
    "../maps/spacex_interactive_map.html"
)

print("Interactive Folium map saved successfully.")
