import ee

# You said you've already authenticated; typically you just need:
# ee.Initialize() 
# If you use a specific billing project / GCP project:
# ee.Initialize(project="YOUR_GCP_PROJECT_ID")

ee.Initialize(project = 'alpha-earth-test-483513')

dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

lon, lat = 2.3522, 48.8566  # Paris example
pt = ee.Geometry.Point([lon, lat])

year = 2023
img = (dataset
       .filterDate(f"{year}-01-01", f"{year+1}-01-01")
       .filterBounds(pt)
       .first())

bands = [f"A{i:02d}" for i in range(64)]

# sample the pixel value at the point
d = img.select(bands).reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=pt,
    scale=10,          # dataset is 10m
    maxPixels=1e9
).getInfo()

embedding = [d[b] for b in bands]  # 64 floats
print(embedding[:5], len(embedding))

