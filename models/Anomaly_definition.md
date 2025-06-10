### Anomaly Definition
By definion anomaly is a deviation from the usual result. After analyzing and visualising the data, we reviewed our anomaly definition.
Here is the list of deviations we determined to be anomalies in ship trajectory data after the analysis:

1. Path deviation - a vessel strays too much from the regular path on a given route. (latitute/longitude)
2. Speed deviation - an irregular, significant change of speed occurs for the vessel, at a position at which such change is not expected. (speed_over_ground SOG) 
Note: Speed changes, that have valid reasons e.g. land proximity or high likelyhood of encountering other vessels are not anomalies.
3. Direction deviation - an irregular, significant change of course_over_ground (COG) and true_heading (TH), at a position at which such change is not expected.
Note: Same as speed, changes with valid reasons are not anomalies.
4. Start/End location deviation - a vessel starts its trip not in declared start destination and end destination.
5. Temporary stop deviation - a vessel stops for a long time at an irregular position, without clear reason.