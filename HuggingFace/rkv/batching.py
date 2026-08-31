import torch


def valid_mask_from_lengths(valid_lengths, physical_len):
    """(bsz, physical_len) bool mask, True where the slot holds real content.

    Left-padding invariant: row i's real content occupies the *last* `valid_lengths[i]` slots,
    i.e. every invalid slot precedes every valid slot. Group slicing below relies on this.
    """
    positions = torch.arange(physical_len, device=valid_lengths.device)
    return positions.view(1, -1) >= (physical_len - valid_lengths.view(-1, 1))


def padding_present(valid_lengths, physical_len):
    """True when at least one row is shorter than the physical tensor."""
    return valid_lengths is not None and bool((valid_lengths != physical_len).any())


def build_additive_attention_mask(valid_lengths, physical_len, q_len, dtype, device):
    """Additive float mask of shape (bsz, 1, q_len, physical_len): 0 to attend, finfo.min to not.

    Built from our own validity bookkeeping rather than from HuggingFace's `attention_mask`,
    because after compression HF's version no longer describes the cache layout (reason 2 in
    the module docstring). `finfo(dtype).min` rather than `-inf` so fp16 cannot produce NaN.
    """
    valid = valid_mask_from_lengths(valid_lengths, physical_len)
    allowed = valid.view(-1, 1, 1, physical_len).expand(-1, 1, q_len, physical_len)

    if q_len > 1:
        # query t occupies physical slot (physical_len - q_len + t)
        q_phys = torch.arange(q_len, device=device).view(-1, 1) + (physical_len - q_len)
        k_phys = torch.arange(physical_len, device=device).view(1, -1)
        causal = (k_phys <= q_phys).view(1, 1, q_len, physical_len)
        # A padding query row would otherwise be fully masked; softmax over an all-min row is
        # NaN, and that NaN reaches the *keys* of the next layer, where it contaminates real
        # queries through the mask (NaN + finfo.min is still NaN). Letting every query attend
        # to its own slot keeps such rows finite; it is a no-op for real queries, which already
        # attend to themselves.
        allowed = (allowed & causal) | (k_phys == q_phys).view(1, 1, q_len, physical_len)

    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)


def compress_by_length_group(
    update_fn, valid_lengths, key_states, cached_queries, value_states,
    extras=(), extra_pad_values=(), head_extras=(),
):
    """Run `update_fn` per length-group on padding-free slices, reassemble right-aligned.

    `update_fn(keys, queries, values, *extras, *head_extras) -> (keys, values, *extras)` is the
    kernel's own `update_kv`, called completely unmodified.

    Two metadata channels, because they are indexed differently:
      * `extras` -- PER SLOT, shaped (bsz, n_kv_heads, physical_len). Row-selected *and*
        sequence-sliced alongside the keys, and returned re-indexed to the compressed cache, with
        `extra_pad_values[i]` written into any padding. CovarianceMerge's `beta`/`n`.
      * `head_extras` -- PER (row, KV head), no sequence axis. Only row-selected, never sliced and
        never returned. CovarianceMerge's future-query `mu`/`Sigma`.

    Returns `(outputs_tuple, new_valid_lengths)`.
    """
    bsz, n_kv_heads, physical_len, head_dim = key_states.shape
    device = key_states.device

    # group seuquences by their true lengths
    groups = {}
    for row, length in enumerate(valid_lengths.tolist()):
        groups.setdefault(length, []).append(row)

    # apply compression for each group sequentially.
    per_group = []
    new_lengths = [0] * bsz
    for length, rows in groups.items():
        rows_idx = torch.tensor(rows, device=device)
        start = physical_len - length
        outputs = update_fn(
            key_states[rows_idx][:, :, start:, :],
            cached_queries[rows_idx],
            value_states[rows_idx][:, :, start:, :],
            *[extra[rows_idx][:, :, start:] for extra in extras],
            *[head[rows_idx] for head in head_extras],
        )
        if not isinstance(outputs, tuple):
            raise TypeError("update_kv must return a tuple of tensors")
        per_group.append((rows_idx, outputs))
        for row in rows:
            new_lengths[row] = outputs[0].shape[2]

    out_len = max(new_lengths)
    out_key = key_states.new_zeros(bsz, n_kv_heads, out_len, head_dim)
    out_value = value_states.new_zeros(bsz, n_kv_heads, out_len, head_dim)
    out_extras = [
        extra.new_full((bsz, n_kv_heads, out_len), pad_value)
        for extra, pad_value in zip(extras, extra_pad_values)
    ]

    # prepare compressed key and value to return
    for rows_idx, outputs in per_group:
        group_len = outputs[0].shape[2]
        start = out_len - group_len
        out_key[rows_idx, :, start:, :] = outputs[0]
        out_value[rows_idx, :, start:, :] = outputs[1]
        for out_extra, group_extra in zip(out_extras, outputs[2:]):
            out_extra[rows_idx, :, start:] = group_extra

    new_valid_lengths = torch.tensor(new_lengths, dtype=valid_lengths.dtype, device=device)
    return (out_key, out_value, *out_extras), new_valid_lengths
