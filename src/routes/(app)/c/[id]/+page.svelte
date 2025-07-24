<script lang="ts">
	import { page } from '$app/stores';

	import Chat from '$gclib/components/chat/Chat.svelte';
</script>

<Chat chatIdProp={$page.params.id} />
