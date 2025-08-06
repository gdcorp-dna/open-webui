<script lang='ts'>


import { goto } from '$app/navigation';
import Switch from '$lib/components/common/Switch.svelte';
import ChatBubbleOval from '$lib/components/icons/ChatBubbleOval.svelte';
import { isSDMEnabled,settings,user } from '$lib/stores';
import { getContext } from 'svelte';
const i18n = getContext('i18n');
let show = false;
const handleToggle = async () => {
	isSDMEnabled.set(!$isSDMEnabled);
	await goto('/');
	const newChatButton = document.getElementById('new-chat-button');
	setTimeout(() => {
		newChatButton?.click();
	}, 0);
	show = !show;
};
console.log(settings,user)
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