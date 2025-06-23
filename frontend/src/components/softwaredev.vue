<template>
	<div class="app-container">
		<div class="content">
			<aside class="sidebar">
				<h2>Select Model</h2>
				<select v-model="selectedModel" @change="onModelChange">
					<option disabled hidden value="">-- Choose a model --</option>
					<option v-for="option in modelOptions" :key="option.value" :value="option.value">
						{{ option.label }}
					</option>
				</select>

				<h2>Select a cruise</h2>
				<ul class="trip-list">
					<li
						v-for="trip in trips"
						:key="trip"
						:class="{ active: trip === selectedTripId }"
						@click="selectTrip(trip)">
						{{ trip }}
					</li>
				</ul>

				<button class="train-button" :disabled="!selectedModel || !selectedTripId || loading" @click="runModel">
					{{ loading ? "Running..." : "Run Model" }}
				</button>
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
						<button @click="chartInstance.resetZoom()">Reset zoom</button>
					</div>
					<div class="chart-scroll">
						<canvas v-show="selectedMetric !== 'all_points'" id="metric-chart"> </canvas>

						<div v-show="selectedMetric === 'all_points'" class="points-list">
							<div
								v-for="(pt, idx) in tripData"
								:key="idx"
								@click="handlePointClick(idx)"
								:class="[
									'point-item',
									pt.is_anomaly_pred ? 'anomaly' : 'normal',
									idx === selectedPointIdx ? 'active' : '',
								]">
								<strong>#{{ idx + 1 }}</strong>
								Lat: {{ pt.latitude }}, Lon: {{ pt.longitude }}, Score: {{ pt.score }}
							</div>
						</div>
					</div>
				</section>
			</main>
		</div>
	</div>
</template>

<script setup>
	import { ref, onMounted, watch, nextTick } from "vue";
	import Chart from "chart.js/auto";
	import zoomPlugin from "chartjs-plugin-zoom";
	import L from "leaflet";
	import "leaflet/dist/leaflet.css";

	import localTrips from "@/assets/trips.json";

	Chart.register(zoomPlugin);
	// Base URL for backend API
	const endpoint = "debug";

	// State
	const trips = ref([]);
	const selectedTripId = ref(null);
	const selectedModel = ref("");
	const loading = ref(false);
	const tripData = ref([]);
	let chartInstance = null;

	// Options
	const modelOptions = [
		{ value: "1", label: "OC-SVM" },
		{ value: "2", label: "Isolation Forest" },
		{ value: "3", label: "Logistic Regression" },
		{ value: "4", label: "Random Forest" },
	];
	const metricOptions = ref([]);
	const selectedMetric = ref("");

	let map = null;
	let markersGroup = null;
	let polyline = null;
	const selectedPointIdx = ref(null);
	const markerRefs = [];

	// Draw points on the map
	function drawPoints(points, options = { showIndices: false }) {
		const { showIndices } = options;

		if (markersGroup) markersGroup.clearLayers();
		if (polyline) polyline.remove();

		const latlngs = points.map((pt) => [pt.latitude, pt.longitude]);
		markersGroup = L.layerGroup().addTo(map);

		markerRefs.length = 0;

		points.forEach((pt, idx) => {
			const baseColor = pt.is_anomaly_pred ? "red" : "blue";
			const marker = L.circleMarker([pt.latitude, pt.longitude], {
				radius: 6,
				color: baseColor,
				fillColor: baseColor,
				fillOpacity: 0.8,
				weight: 1,
			})
				.addTo(markersGroup)
				.on("click", () => handlePointClick(idx));

			markerRefs.push(marker);

			if (options.showIndices) {
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

	// Fetch list of trips from backend
	async function fetchTrips() {
		if (endpoint == "debug") {
			trips.value = localTrips;
			return;
		}

		try {
			const res = await fetch(`${endpoint}/gettrips`);
			trips.value = await res.json();
		} catch (err) {
			console.error("Failed to fetch trips", err);
		}
	}

	// Fetch and render trip data
	async function fetchTripData() {
		if (!selectedModel.value || !selectedTripId.value) return;
		loading.value = true;

		let data;
		try {
			if (endpoint === "debug") {
				const module = await import("@/assets/trip.json");
				data = module.default;
			} else {
				const res = await fetch(`${endpoint}/train?model=${selectedModel.value}&trip=${selectedTripId.value}`);
				data = await res.json();
			}

			if (Array.isArray(data) && data.length > 0 && data[0].data) {
				const keys = Object.keys(data[0].data);
				metricOptions.value = keys.map((key) => ({
					value: key,
					label: key,
				}));
				metricOptions.value.push({ value: "all_points", label: "All Points" });
				selectedMetric.value = metricOptions.value[0].value;
			}

			drawPoints(data);
			tripData.value = data;
			selectedPointIdx.value = null;

			createChart();
		} catch (err) {
			console.error("Failed to fetch trip data", err);
		} finally {
			loading.value = false;
		}
	}

	function selectTrip(id) {
		selectedTripId.value = id;
	}

	function onModelChange() {}

	async function runModel() {
		await fetchTripData();
	}

	function createChart() {
		if (selectedMetric.value === "all_points") return;

		const metric = selectedMetric.value;
		if (!metric || tripData.value.length === 0) return;

		const labels = tripData.value.map((pt, idx) =>
			pt.time_stamp ? new Date(pt.time_stamp).toLocaleString() : `#${idx + 1}`
		);
		const values = tripData.value.map((pt) => {
			const v = pt.data?.[metric];
			return typeof v === "number" ? v : null;
		});

		if (chartInstance) {
			chartInstance.destroy();
		}

		const ctx = document.getElementById("metric-chart").getContext("2d");
		chartInstance = new Chart(ctx, {
			type: "line",
			data: {
				labels,
				datasets: [
					{
						label: metric,
						data: values,
						fill: false,
						tension: 0.1,
						pointRadius: 3,
					},
				],
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				devicePixelRatio: window.devicePixelRatio || 1,
				onClick: (evt, elements) => {
					if (elements.length) {
						const idx = elements[0].index;
						handlePointClick(idx);
					}
				},
				scales: {
					x: {
						display: true,
						title: { display: true, text: "Time" },
					},
					y: {
						display: true,
						title: { display: true, text: metric },
					},
				},
				plugins: {
					legend: { display: false },
					zoom: {
						pan: {
							enabled: true,
							mode: "x",
							threshold: 5,
						},
						zoom: {
							wheel: {
								enabled: true,
							},
							pinch: {
								enabled: true,
							},
							mode: "x",
							onZoom: ({ chart }) => {},
						},
					},
				},
			},
		});
	}

	function handlePointClick(idx) {
		selectedPointIdx.value = idx;
		updateMarkerColors();
		highlightOnChart();
		highlightInList();
	}

	function updateMarkerColors() {
		markerRefs.forEach((m, i) => {
			const base = tripData.value[i].is_anomaly_pred ? "red" : "blue";
			const color = i === selectedPointIdx.value ? "green" : base;
			m.setStyle({ color, fillColor: color });
			if (i === selectedPointIdx.value) {
				map.flyTo(m.getLatLng(), map.getZoom(), { duration: 0.4 });
			}
		});
	}

	function highlightOnChart() {
		if (selectedMetric.value === "all_points" || !chartInstance) return;

		const i = selectedPointIdx.value;
		const ds = chartInstance.data.datasets[0];

		if (!Array.isArray(ds.pointBackgroundColor)) {
			ds.pointBackgroundColor = ds.data.map((_, k) => (tripData.value[k].is_anomaly_pred ? "red" : "blue"));
		}

		ds.pointBackgroundColor = ds.pointBackgroundColor.map((c, k) =>
			k === i ? "green" : tripData.value[k].is_anomaly_pred ? "red" : "blue"
		);

		const min = Math.max(i - 5, 0);
		const max = Math.min(i + 5, ds.data.length - 1);
		chartInstance.resetZoom();
		chartInstance.zoomScale("x", { min, max });

		chartInstance.update("none");
	}

	function highlightInList() {
		if (selectedMetric.value !== "all_points") return;
		nextTick(() => {
			const listEl = document.querySelector(".points-list");
			const item = listEl?.children[selectedPointIdx.value];
			if (item) {
				item.scrollIntoView({ block: "center", behavior: "smooth" });
			}
		});
	}

	watch(selectedMetric, () => {
		selectedPointIdx.value = null;

		createChart();
	});

	onMounted(async () => {
		map = L.map("map").setView([53.57, 8.53], 6);
		L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
			maxZoom: 18,
		}).addTo(map);
		markersGroup = L.layerGroup().addTo(map);

		await fetchTrips();
	});
</script>

<style scoped lang="scss">
	@use "../style/style.scss" as *;
	@import "leaflet/dist/leaflet.css";

	.app-container {
		display: flex;
		flex-direction: column;
		height: 800px;
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

	.sidebar select,
	.sidebar .train-button {
		width: 100%;
		padding: 0.5rem;
		margin-bottom: 1rem;
	}

	.trip-list {
		list-style: none;
		max-height: 700px;
		overflow-y: scroll;
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
		height: 800px;
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

	.chart-scroll {
		display: block;
		height: 800px;
		overflow-x: scroll;
		overflow-y: hidden;
		background-color: #f0f0f0;
		border: 1px solid #ccc;
		max-width: 300px;
	}

	.chart-scroll canvas {
		display: block;
		width: 800px;
		height: 800px;
	}

	.train-button {
		background-color: #007bff;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	.train-button:disabled {
		background-color: #ccc;
		cursor: not-allowed;
	}

	.points-list {
		max-height: 800px;
		overflow-y: auto;
		padding: 0.5rem;
		background: #f9f9f9;
	}
	.point-item {
		padding: 0.5rem;
		margin-bottom: 0.25rem;
		color: #fff;
		border-radius: 4px;
	}
	.point-item.normal {
		background-color: blue;
	}
	.point-item.anomaly {
		background-color: red;
	}
	.point-item.active {
		background-color: green !important;
	}
</style>
