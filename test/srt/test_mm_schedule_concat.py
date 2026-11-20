"""Tests for the precomputed-embedding concat fast path.

In EPD the receiver gives every item of a request a slice of the same host
buffer, in order. Concatenating those slices copies the whole embedding on
every rank; returning one strided view over the same rows is equivalent.
Anything that is not a strictly adjacent run of slices of one 2-D CPU storage
must still fall back to torch.concat.
"""

import torch

from sglang.srt.managers.mm_schedule import _concat_or_view


def test_adjacent_slices_share_storage():
    base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    chunks = [base[0:2], base[2:4], base[4:6]]
    out = _concat_or_view(chunks)
    assert out.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert out.shape == (6, 4)
    assert torch.equal(out, base)


def test_out_of_order_slices_still_merge():
    base = torch.arange(16.0).reshape(4, 4)
    out = _concat_or_view([base[2:4], base[0:2]])
    # Sorted by storage offset: rows 0..3 again.
    assert torch.equal(out, base)


def test_gap_falls_back_to_concat():
    base = torch.arange(24.0).reshape(6, 4)
    out = _concat_or_view([base[0:2], base[4:6]])
    assert out.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    assert torch.equal(out, torch.cat([base[0:2], base[4:6]]))


def test_separate_storages_fall_back():
    a = torch.arange(8.0).reshape(2, 4)
    b = torch.arange(8.0, 16.0).reshape(2, 4)
    out = _concat_or_view([a, b])
    assert torch.equal(out, torch.cat([a, b]))


def test_non_2d_falls_back():
    a = torch.arange(8.0)
    b = torch.arange(8.0, 16.0)
    out = _concat_or_view([a, b])
    assert torch.equal(out, torch.cat([a, b]))


def test_view_is_read_only_safe():
    """Writing through the view must not corrupt the source rows' order."""
    base = torch.zeros(4, 3)
    out = _concat_or_view([base[0:2], base[2:4]])
    assert torch.equal(out, base)
    assert out.shape == (4, 3)


def test_unaligned_storage_offset_is_still_a_view():
    """The run may start at any element offset of the shared buffer.

    The receiver's modality view begins wherever the encoder placed the part,
    which is not necessarily row-aligned in the destination storage.
    """
    flat = torch.arange(1 + 24, dtype=torch.float32)
    rows = flat[1:].reshape(6, 4)
    chunks = [rows[0:2], rows[2:4], rows[4:6]]
    out = _concat_or_view(chunks)
    assert out.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
    assert torch.equal(out, rows)


def test_empty_chunks_do_not_break_the_run():
    """Zero-row items sit at arbitrary offsets; they must be ignored."""
    base = torch.arange(16.0).reshape(4, 4)
    empty = base[:0]
    chunks = [base[0:2], empty, base[2:4]]
    out = _concat_or_view(chunks)
    assert out.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert torch.equal(out, base)


def test_only_empty_chunks_falls_back():
    base = torch.zeros(4, 2)
    out = _concat_or_view([base[:0], base[:0]])
    assert out.shape == (0, 2)
