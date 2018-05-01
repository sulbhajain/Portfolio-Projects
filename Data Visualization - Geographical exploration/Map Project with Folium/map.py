import folium
import pandas as  pd

data = pd.read_csv("Volcanoes_USA.txt")

lat = list(data['LAT'])
lon = list(data['LON'])
elev = list(data['ELEV'])

def set_color(elevation):
    if elevation <1000:
        return 'green'
    elif elevation <=3000:
        return 'red'
    else:
        return 'orange'

map1 = folium.Map(location=[47, -122], zoom_start=6, tiles="Mapbox Bright")

fg= folium.FeatureGroup(name="My Map")
for lt, lg, el in zip(lat, lon, elev):
    fg.add_child(folium.Marker(location =[lt, lg], popup=str(el), icon=folium.Icon(set_color(el))))

map1.add_child(fg)

map1.save("Map1.html")
