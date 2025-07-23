<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import Switch from '$lib/components/common/Switch.svelte';
	import { sdmMode, settings, user } from '$lib/stores';
	import { updateUserSettings } from '$lib/apis/users';

	const i18n = getContext('i18n');
	
	// Optional parameter to control whether the component should be displayed
	export let showIfDisabled = false;
	
	// Check if SDM mode is allowed in user settings (default to false if not defined)
	$: sdmAllowed = $settings?.sdmAllowed ?? false;
	
	// Check if user is admin
	$: isAdmin = $user?.role === 'admin';
	
	// Determine if switch should be shown (if explicitly allowed or the user is an admin)
	$: showSwitch = showIfDisabled || sdmAllowed || isAdmin;

	// Initialize SDM mode from user settings if available
	onMount(() => {
		if ($settings && $settings.sdmMode !== undefined) {
			sdmMode.set($settings.sdmMode);
		}
	});

	function handleToggle() {
		// Toggle SDM mode
		sdmMode.set(!$sdmMode);

		// Save to user settings
		if (localStorage.token && $settings) {
			// Update settings store first
			settings.update(currentSettings => {
				return { ...currentSettings, sdmMode: $sdmMode };
			});
			// Then send to server
			updateUserSettings(localStorage.token, { ui: $settings });
		}
	}
</script>

{#if showSwitch}
	<Switch 
		state={$sdmMode} 
		on:change={handleToggle}
	/>
{/if}