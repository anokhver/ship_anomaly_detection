<template>
	<div class="app-container">
		<div class="content">
			<aside class="sidebar">
				<h2>Select a cruise</h2>
				<ul class="trip-list">
					<li
						v-for="trip in trips"
						:key="trip.id"
						:class="{ active: trip.id === selectedTripId }"
						@click="selectTrip(trip.id)">
						{{ trip.name }}
					</li>
				</ul>
			</aside>
			<main class="main-view">
				<section class="map-view">
					<div id="map" class="map-container"></div>
				</section>
				<section class="chart-view">
					<div class="chart-controls">
						<label for="metric-select">Select metrics:</label>
						<select id="metric-select" v-model="selectedMetric">
							<option v-for="option in metricOptions" :key="option.value" :value="option.value">
								{{ option.label }}
							</option>
						</select>
					</div>
					<div class="chart-placeholder">
						<p v-if="!selectedTripId">Select a cruise to see the chart</p>
						<p v-else>Chart: {{ selectedMetric }} for trip ID {{ selectedTripId }}</p>
					</div>
				</section>
			</main>
		</div>
	</div>
</template>

<script setup>
	import { onMounted } from "vue";
	import L from "leaflet";
	import "leaflet/dist/leaflet.css";

	import tripData from "@/assets/trip.json";

	let map = null;
	let markersGroup = null;
	let polyline = null;

	function drawPoints(points, options = { showIndices: false }) {
		const { showIndices } = options;

		if (markersGroup) {
			markersGroup.clearLayers();
		}
		if (polyline) {
			polyline.remove();
		}

		const latlngs = points.map((pt) => [pt.latitude, pt.longitude]);

		markersGroup = L.layerGroup().addTo(map);

		points.forEach((pt, idx) => {
			const color = pt.is_anomaly_pred ? "red" : "blue";

			const marker = L.circleMarker([pt.latitude, pt.longitude], {
				radius: 6,
				color: color,
				fillColor: color,
				fillOpacity: 0.8,
				weight: 1,
			}).addTo(markersGroup);

			if (showIndices) {
				marker.bindTooltip(`${idx + 1}`, {
					permanent: true,
					direction: "right",
					className: "point-index-tooltip",
				});
			}
		});

		if (latlngs.length) {
			map.fitBounds(L.latLngBounds(latlngs));
		}
	}

	onMounted(() => {
		map = L.map("map").setView([53.57, 8.53], 6);
		L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
			maxZoom: 18,
			attribution: "&copy; OpenStreetMap contributors",
		}).addTo(map);

		markersGroup = L.layerGroup().addTo(map);

		drawPoints(tripData);
	});
</script>

<style scoped lang="scss">
	@use "../style/style.scss" as *;
	@import "leaflet/dist/leaflet.css";

	.app-container {
		display: flex;
		flex-direction: column;
		min-height: 800px;
	}

	.content {
		display: flex;
		flex: 1;
	}

	.sidebar {
		width: 250px;
		padding: 1rem;
		border-right: 1px solid #ccc;
	}

	.sidebar h2 {
		margin-bottom: 0.5rem;
		font-size: 1.1rem;
	}

	.trip-list {
		list-style: none;
		padding: 0;
		margin: 0;
	}

	.trip-list li {
		padding: 0.5rem;
		cursor: pointer;
		border-radius: 4px;
	}

	.trip-list li.active,
	.trip-list li:hover {
		background-color: #d0d0d0;
	}

	.main-view {
		flex: 1;
		display: grid;
		grid-template-columns: 2fr 1fr;
		gap: 1rem;
		padding: 1rem;
	}

	.map-view {
		position: relative;
	}

	.map-container {
		width: 100%;
		height: 100%;
		min-height: 400px;
	}

	.chart-view {
		display: flex;
		flex-direction: column;
	}

	.chart-controls {
		margin-bottom: 0.5rem;
		display: flex;
		align-items: center;
	}

	.chart-controls label {
		margin-right: 0.5rem;
	}

	.chart-placeholder {
		flex: 1;
		background-color: #f0f0f0;
		border: 1px solid #ccc;
		display: flex;
		justify-content: center;
		align-items: center;
		color: #666;
		font-size: 1rem;
	}
</style>
