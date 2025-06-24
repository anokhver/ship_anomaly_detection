# Model features
This is a summary of the data passed to the models during training. It is derived from the original data (after cleaning).

- **speed_over_ground**
    - Speed of the vessel at current time, SOG value from the source data
- **dv**
    - Delta of velocity - difference of speed between the current point and the previous one, derived from SOG
- **dcourse**
    - Delta of the course - difference of COG between the current point and the previous one, adjusted for angle calculations
- **ddraft**
    - Delta of the draught - difference of draught between the current point and the previous one
- **zone**
    - A pre-defined coordinate range, for which the models can differentiate between the "port" and "open sea" zones - in ports anomalous behaviour is supposed to be treated less harshly
- **x_km**
    - A substitute for evaluating coordinates - a "center of the route is established based on the coordinates of all entries in that route. Then, the distance from that center is calculated for an entry based on longitude difference and converted to kilometers.
- **y_km**
    - Same as x_km, only calculating latitude difference.
- **dist_to_ref**
    - Distance for each point to the average route. Average trajectory for a route is computed by resampling each trip to n_points along cumulative distance fraction, then averaging. 
- **route_dummy**
    - Scrapped idea, replaced by dispatcher files. Will be removed in sprint 4.