// frontend/src/router/index.js
import { createRouter, createWebHistory } from "vue-router";
import softwaredev from "@/components/softwaredev.vue";

const routes = [
	{
		path: "/",
		name: "softwaredev",
		component: softwaredev,
	},
];

const router = createRouter({
	history: createWebHistory(import.meta.env.BASE_URL),
	routes,
	scrollBehavior(to, from, savedPosition) {
		return savedPosition || { top: 0 };
	},
});

export default router;
