<template>
	<div id="container-main">
		<div id="main-1">
			<router-view />
		</div>
	</div>
</template>
<script>
	export default {};
</script>
<style lang="scss" scoped>
	@use "../style/style.scss" as *;
	#container-main {
		display: flex;
		justify-content: center;

		#main-1 {
			width: 100vw;
			max-width: 1400px;
		}
	}
</style>
