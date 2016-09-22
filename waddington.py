import numpy as np
import itertools
from getch import pause

possible_blocks = np.asarray([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], np.int8)


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
        raise ValueError("bad position")

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
        raise ValueError("bad layer")

def generate_pin_rows(size, offset=0):
    # Pins can be offset by 0, 1 or 2
    if offset > 2 or offset < 0:
        raise ValueError("bad offset")
    
    # There is a pin every 3 holes
    num_pins = (size - offset + 2) // 3
    for x in range(2 ** num_pins):
        layer = np.zeros(size, np.int8)
        pos = offset
        # We simply use the binary encoding of an integer to generate the
        # combinations
        for i in xrange(num_pins -1, -1, -1):
            layer[pos] = (x >> i) & 1 
            pos += 3
        yield layer

def generate_layers(size, num, layers=[]):
    # Create the generators for each layer
    off = 0
    gen_all = []
    for i in range(num):
        gen = generate_pin_rows(size, off)
        gen_all.append(gen)
        off = (off + 2) % 3

    for layers in itertools.product(*gen_all):
        yield layers

def generate_paths(layers, pos, cur=0):
    if cur == len(layers):
        yield pos
    else:
        for newpos in move_through_layer(pos, 1, layers[cur]):
            for finalpos in generate_paths(layers, newpos, cur+1):
                yield finalpos

def test1():
    layer = np.zeros(5, np.int8)
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

def test2():
    count = 0
    for k in generate_layers(15, 3):
        count += 1
    print count

def test3():
    for layers in generate_layers(9, 6):
        print 'new-layout---'
        for l in layers:
            print l
        for finalpos in generate_paths(layers, 4):
            print finalpos
        pause()

def test4():
    dist = dict([(i, 0) for i in range(9)])
    for layers in generate_layers(9, 6):
        for finalpos in generate_paths(layers, 4):
            dist[finalpos] += 1
    print dist

if __name__ == '__main__':
    test4()

    



