# GCLib - Custom Code Contribution Pattern

This directory (`src/gclib`) contains our custom modifications to the Open WebUI codebase. The pattern described below allows us to maintain our custom changes while minimizing conflicts when merging updates from the original repository.

## Pattern for Custom Contributions

1. We've created an alias `$gclib` which points to the `src/gclib` folder.

2. When modifying code within the `$lib` alias (which points to `src/lib`):

   - Make a copy of the file and place it under the `src/gclib` folder using the same path structure
   - For example, if modifying `src/lib/components/common/Example.svelte`, create `src/gclib/components/common/Example.svelte`

3. Replace imports used within the original `$lib` files to instead use the `$gclib` versions:

   - Example: Change `import Example from '$lib/components/common/Example.svelte'` to `import Example from '$gclib/components/common/Example.svelte'`

4. In your custom `$gclib` files, continue to use `$gclib` imports for custom components and `$lib` imports for unmodified components.

## Benefits

This approach:

- Isolates all custom code to the `$gclib` directory
- Requires minimal changes to the original `$lib` folder (only import statements)
- Makes it easier to identify and maintain our custom modifications
- Reduces merge conflicts when pulling updates from the original repository
