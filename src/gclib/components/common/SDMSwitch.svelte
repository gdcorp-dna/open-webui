<script lang='ts'>
import { goto } from '$app/navigation';
import Switch from '$lib/components/common/Switch.svelte';
import ChatBubbleOval from '$lib/components/icons/ChatBubbleOval.svelte';
import { isSDMEnabled, settings, user } from '$lib/stores';
import { getContext, createEventDispatcher, onMount } from 'svelte';

const i18n = getContext('i18n');
const dispatch = createEventDispatcher();

// Local storage key for SDM enabled state
const SDM_ENABLED_KEY = 'sdmEnabled';

let show = false;

// Load the stored state on component mount or set to disabled by default
onMount(() => {
	const storedValue = localStorage.getItem(SDM_ENABLED_KEY);
	// If there's a stored value, use it; otherwise ensure it's disabled by default
	isSDMEnabled.set(storedValue === 'true'); // This will evaluate to false when storedValue is null
	
	// Ensure the localStorage reflects the default disabled state if no value was previously set
	if (storedValue === null) {
		localStorage.setItem(SDM_ENABLED_KEY, 'false');
	}
});

const handleToggle = async () => {
	const newValue = !$isSDMEnabled;
	
	// Update the store value
	isSDMEnabled.set(newValue);
	
	// Save to localStorage for persistence
	localStorage.setItem(SDM_ENABLED_KEY, newValue.toString());
	
	// Dispatch event when toggled to filter models with SDM tag
	dispatch('change', { filterSDM: newValue });
	
	// Navigate to home and create a new chat
	await goto('/');
	
	// Find and click the new chat button after the navigation completes
	const newChatButton = document.getElementById('new-chat-button');
	if (newChatButton) {
		setTimeout(() => newChatButton.click(), 0);
	}
	
	// Toggle local show state
	show = !show;
};
</script>
<div class="flex items-center mx-2 mt-1 mb-2">
	<button
		class="flex justify-between w-full font-medium line-clamp-1 select-none items-center rounded-button py-2 px-3 text-sm text-gray-700 dark:text-gray-100 outline-hidden transition-all duration-75 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg cursor-pointer data-highlighted:bg-muted"
		on:click={handleToggle}
	>
		<div class="flex gap-2.5 items-center">
			<ChatBubbleOval className="size-4" strokeWidth="2.5" />

			{$i18n.t(`SDM Mode`)}
		</div>

		<div>
			<Switch state={$isSDMEnabled} />
		</div>
	</button>
</div>