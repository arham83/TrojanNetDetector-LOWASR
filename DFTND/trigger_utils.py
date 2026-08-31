"""Trigger functions shared by training, evaluation, and reverse training."""

import torch


def trigger_locations(position, method='pattern'):
    row, col = map(int, position)
    if method == 'pixel':
        return ((row, col),)
    if method in ('pattern', 'pattern2'):
        return (
            (row, col), (row + 1, col + 1), (row - 1, col + 1),
            (row + 1, col - 1), (row - 1, col - 1),
        )
    if method == 'ell':
        return ((row, col), (row + 1, col), (row, col + 1))
    if method == 'squre':
        return tuple(
            (row + dr, col + dc)
            for dr in range(-3, 4) for dc in range(-3, 4)
        )
    raise ValueError('Unsupported tensor trigger method {!r}'.format(method))


def add_trigger_tensor(images, position, color, method='pattern'):
    """Apply a configured trigger to a CHW image or NCHW batch in [0, 1]."""
    if images.ndim not in (3, 4):
        raise ValueError('Expected CHW or NCHW input, got {}'.format(tuple(images.shape)))
    output = images.clone()
    locations = trigger_locations(position, method)
    height, width = output.shape[-2:]
    if any(not (0 <= r < height and 0 <= c < width) for r, c in locations):
        raise ValueError('Configured trigger extends outside the image')
    value = torch.as_tensor(
        color, dtype=output.dtype, device=output.device
    ).flatten().div(255.0)
    channels = output.shape[-3]
    if value.numel() == 1:
        value = value.expand(channels)
    elif value.numel() != channels:
        raise ValueError(
            'Trigger color has {} values but input has {} channels'.format(
                value.numel(), channels
            )
        )
    for row, col in locations:
        if output.ndim == 3:
            output[:, row, col] = value
        else:
            output[:, :, row, col] = value.view(1, channels)
    return output
