<template>
	<header class="app-header">
		<h1>Maritime AI</h1>
	</header>
	<main>
		<mainView />
	</main>
	<footer class="app-footer">
		<p>&copy; 2025 Maritime AI</p>
	</footer>
</template>

<script>
	import mainView from "./components/mainView.vue";
	export default {
		name: "App",
		components: {
			mainView,
		},
	};
</script>
<style lang="scss">
	* {
		box-sizing: border-box;
		font-family: foncik;
	}
	#app {
		font-family: foncik;
		-webkit-font-smoothing: antialiased;
		-moz-osx-font-smoothing: grayscale;
		color: $font-color;
		box-sizing: border-box;
	}
	body {
		margin: 0;
		background-color: $main-color;
	}

	.app-header,
	.app-footer {
		background-color: $main-color;
		padding: 1rem;
		text-align: center;
	}
</style>
