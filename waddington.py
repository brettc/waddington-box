import numpy as np

possible_blocks = np.asarray([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], bool)


def maybe_bounce(position, size):
    if position < 1:
        return -position + 1
    elif position >= size:
        over = size - position + 1
        return size - over

    return position

def move_through_layer(position, momentum, layer):
    # We assume our ball has width, so the position cannot be right on the
    # edges!
    size = len(layer)
    if position < 1 or position >= size-1:
        raise ValueError

    # Get the what's in front of the ball
    block = layer[position-1: position+2]
    # print block
    # print possible_blocks[1]

    # Now the rules
    if (block == possible_blocks[0]).all():
        yield maybe_bounce(position, size)
    elif (block == possible_blocks[1]).all():
        yield maybe_bounce(position + 1, size)
        yield maybe_bounce(position - 1, size)
    elif (block == possible_blocks[2]).all():
        yield maybe_bounce(position + momentum, size)
    elif (block == possible_blocks[3]).all():
        yield maybe_bounce(position - momentum, size)
    else:
        raise ValueError("bad block")


if __name__ == '__main__':
    layer = np.zeros(5, bool)
    layer[1] = 1
    layer2 = np.roll(layer, 1)
    layer3 = np.roll(layer2, 1)
    print layer
    print layer3
    print layer2
    for p in move_through_layer(1, 1, layer):
        for q in move_through_layer(p, 1, layer3):
            for r in move_through_layer(q, 1, layer2):
                print r
    



