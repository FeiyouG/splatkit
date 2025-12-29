def normalize_batch_tensors(*tensors, spatial_ndim: int):
    """
    Normalize tensors so that all have compatible batch dimensions.
    If a tensor is None, it is returned as is.
    If a tensor's shape is not compatible with the other tensors, an error is raised.
    if a tensor's shape is compatible yet a dimension is singleton, the tensor is expanded to match the longest prefix.

    Args:
        tensors: any number of tensors or None
        spatial_ndim: number of trailing dims considered spatial (e.g. 3 for HWC)

    Returns:
        tuple of tensors with broadcasted batch dimensions
    """

    # Filter out None
    valid = [t for t in tensors if t is not None]
    if not valid:
        return tensors

    # Extract batch shapes
    batch_shapes = []
    for t in valid:
        if t.ndim < spatial_ndim:
            raise ValueError(
                f"Tensor shape {t.shape} has fewer dims than spatial_ndim={spatial_ndim}"
            )
        batch_shapes.append(t.shape[:-spatial_ndim])

    # Determine target batch shape (longest)
    ref_batch = max(batch_shapes, key=len)

    # Validate broadcast compatibility
    for b in batch_shapes:
        if len(b) > len(ref_batch):
            raise ValueError(f"Incompatible batch dims: {b} vs {ref_batch}")

        # Align from the right
        for x, y in zip(reversed(b), reversed(ref_batch)):
            if x != y and x != 1:
                raise ValueError(
                    f"Cannot broadcast batch dims {b} -> {ref_batch}"
                )

    # Expand tensors
    normalized = []
    for t in tensors:
        if t is None:
            normalized.append(None)
            continue

        b = t.shape[:-spatial_ndim]
        if b != ref_batch:
            expand_shape = list(ref_batch) + list(t.shape[-spatial_ndim:])
            t = t.expand(*expand_shape)

        normalized.append(t)

    return tuple(normalized)