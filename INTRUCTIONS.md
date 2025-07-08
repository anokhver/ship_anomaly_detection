# How to use the web application

## Overview

The web interface has a few functions. It marks anomalies red, normal points blue and selected point green. You can:

- Load data
- Select a model
- Choose metrics
- Select a cruise
- Adjust the map zoom
- Reset the chart zoom
- Run the model
- Move and zoom map and chart with the mouse
- Interact with points on the map, list, and charts

---

## Features

### Load data

The application is pre-loaded with our data. If you want to evaluate your own data, use the "Load data" section displayed in the application window and choose the data file you want to upload.
- Must be a CSV in the same format as the original data!
- Large files may take a moment to upload!

After the data is successfully loaded and cleaned. New trip id's will be displayed in the "Select a cruise" section. The upload might fail if an improperly formatted file is used.

### Evaluating the trips

Choose one of the five available models:
1. One-Class SVM
2. Isolation Forest
3. Logistic Regression
4. Random Forest
5. LSTM

Select a trip from the list and then press the "run model" button to evaluate it.

### Metrics

To analyze the data, you can display it according to different metrics on the chart to the right side of the map. The colour-coding of anomalies and non anomalies is consistent with the map, and interacting with a point on either the map or the chart will also mark it on the other.  

Select which metrics you want to see on the chart:

- **Ship type**
  What ship type is there currently. It may change during the trip (e.g., when loading or unloading cargo).
- **Length**
  The length of the ship.
- **Breadth**
  The breadth (width) of the ship.
- **Draught**
  Current draught (vertical distance between the waterline and the bottom of the hull).
- **Speed Over Ground**
  Current speed of the ship.
- **Course Over Ground**
  Current course (direction) of the ship.
- **True Heading**
  Current true heading (orientation) of the ship.
- **Delta Velocity**
  Difference in velocity between the previous and current point.
- **Delta Course**
  Difference in course between the previous and current point.
- **Delta Draught**
  Difference in draught between the previous and current point.
- **Zone**
  `0` if currently in port, `1` if on open sea.
- **X Kilometers**
  Horizontal distance (X axis) from the trip’s start/reference.
- **Y Kilometers**
  Vertical distance (Y axis) from the trip’s start/reference.
- **Distance To Ref**
  Distance from the referential path.
- **All Points**
  A list of all points with their coordinates and model score.


### Adjust map zoom

Use the zoom slider to set the map’s zoom level.

### Reset chart zoom

If you’ve zoomed in too far on the chart, click this button to zoom back out.


---

## Interaction

- **Map Chart/List**
  - Clicking a point on the map will highlight the corresponding chart point and scroll the list to it.
  - Clicking a chart point or list item will zoom the map to that point.

- **Zoom & Pan**
  - The map and chart are both zoomable and pannable with the mouse.
  - The points list is scrollable.

---
