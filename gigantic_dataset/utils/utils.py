import zarr


def merge_zarr_and_subsample(zarr_files, output_array_path, num_samples):
    merged_store = zarr.DirectoryStore(path=f"{output_array_path}.zarr")
    merged_group = zarr.open_group(merged_store, mode='a')

    print(f"Merging zarr groups start")
    # merged_store = zarr.MemoryStore()  # In-memory store for the merged data
    # merged_group = zarr.group(store=merged_store)  # Create the in-memory group

    # Loop over arrays in the first group (assuming all groups have the same array names)
    for array_name in zarr_files[0].array_keys():
        print(f"Merging group: {array_name}")
        # Initialize a list to store the shapes and data arrays from all groups
        total_samples = 0
        arrays = []

        # Collect arrays and calculate total number of samples
        for group in zarr_files:
            # print(f"Reading zarr: {group}")
            arr = group[array_name]
            arrays.append(arr)
            total_samples += arr.shape[0]

            # Check that the arrays have the same shape (except for the first dimension)
            if arr.shape[1:] != arrays[0].shape[1:]:
                raise ValueError(f"Arrays {array_name} have incompatible shapes.")

        # Create the merged array in memory
        first_shape = arrays[0].shape[1:]  # Shape of each sample (excluding the sample dimension)
        dtype = arrays[0].dtype  # Assuming dtype is the same for all arrays
        merged_array = merged_group.create_dataset(array_name, shape=(total_samples, *first_shape), dtype=dtype)

        # Concatenate the arrays from all groups into the merged array
        start = 0
        for arr in arrays:
            end = start + arr.shape[0]
            merged_array[start:end] = arr[:]
            start = end

    output_store = zarr.ZipStore(path=f"{output_array_path}.zip", mode='w')  # Create the output zip store
    output_group = zarr.group(store=output_store)  # Create the output group in the zip store

    print(f"Subsampling zarr groups start")
    # Iterate over each array in the merged group
    for array_name in merged_group.array_keys():

        print(f"Processing group: {array_name}")

        merged_array = merged_group[array_name]

        # Create the output array with the sampled data
        output_array = output_group.create_dataset(array_name, shape=(num_samples, *merged_array.shape[1:]),
                                                   dtype=merged_array.dtype)
        output_array[:] = merged_array[:num_samples]

    # Validate both zarr files have the same attributes and copy attrs to output file
    # assert dict(zarr_files[0].attrs) == dict(zarr_files[1].attrs), "attrs do not match"
    output_group.attrs.put(dict(zarr_files[0].attrs))

    # Close the store to ensure it's properly written to disk
    merged_store.close()
    output_store.close()
