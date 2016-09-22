"""

TODO: Check the bouncing on the right wall

"""
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
    overlap_left = position - 1
    if overlap_left < 0:
        return -overlap_left
    overlap_right = position + 2 - size 
    if overlap_right > 0:
        return size - overlap_right - 1

    return position

def test_maybe_bounce():
    assert maybe_bounce(0, 3) == 1
    assert maybe_bounce(-1, 3) == 2
    assert maybe_bounce(2, 3) == 1
    assert maybe_bounce(3, 3) == 0


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
        yield maybe_bounce(position + 2, size)
        yield maybe_bounce(position - 2, size)
    elif (block == possible_blocks[2]).all():
        yield maybe_bounce(position + 1, size)
    elif (block == possible_blocks[3]).all():
        yield maybe_bounce(position - 1, size)
    else:
        raise ValueError("bad layer")

def generate_rows_1(size, offset=0):
    # Pins can be offset by 0, 1 or 2
    if offset > 3 or offset < 0:
        raise ValueError("bad offset")
    
    # There is a pin every 3 holes
    num_pins = (size - offset + 1) // 4
    for x in range(2 ** num_pins):
        layer = np.zeros(size, np.int8)
        pos = offset
        # We simply use the binary encoding of an integer to generate the
        # combinations
        for i in xrange(num_pins -1, -1, -1):
            layer[pos] = (x >> i) & 1 
            pos += 4
        yield layer


def is_valid_layer(layer, size):
    # Test 1: at least size/4 pins
    pin_count = layer.sum()
    if pin_count < size // 4:
        return False

    # Test 2: pins stand alone, and have gaps of 3
    for i in range(size-3):
        window = layer[i:i+4]
        if window.sum() > 1:
            return False

    return True


def generate_rows_2(size):
    # There is a pin every 3 holes
    for x in range(2 ** size):
        layer = np.zeros(size, np.int8)
        # We simply use the binary encoding of an integer to generate the
        # combinations
        for i in xrange(size -1, -1, -1):
            layer[i] = (x >> i) & 1 

        if is_valid_layer(layer, size):
            yield layer


def generate_layers(size, num, layers=[]):
    # Create the generators for each layer
    off = 0
    gen_all = []
    for i in range(num):
        gen = generate_rows_1(size, off)
        gen_all.append(gen)
        off = (off + 2) % 4

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
    for layers in generate_layers(12, 3):
        print 'new-layout---'
        for l in layers:
            print l
        for finalpos in generate_paths(layers, 5):
            print finalpos
        pause()

def test4():
    dist = dict([(i, 0) for i in range(12)])
    for layers in generate_layers(12, 5):
        for finalpos in generate_paths(layers, 5):
            dist[finalpos] += 1
    print dist

def test5():
    dist = dict([(i, 0) for i in range(7)])
    layers = [
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
    ]
    layers = [np.asarray(l, np.int8) for l in layers]
    for finalpos in generate_paths(layers, 3):
        dist[finalpos] += 1
    print dist

def test6():
    dist = dict([(i, 0) for i in range(17)])
    all = [k for k in generate_rows_2(17)]
    mult = [all] * 5
    for layers in itertools.product(*mult):
        outputs = []
        for finalpos in generate_paths(layers, 8):
            dist[finalpos] += 1
            outputs.append(finalpos)
        if len(outputs) > 7:
            print '-------------------'
            print outputs
            for l in layers:
                print l

    print dist

def test7():
    rows = [r for r in generate_rows_2(17)]
    k = len(rows)
    for i in range(10):
        print i, k
        k = k * k

def test8():
    all = [k for k in generate_rows_2(17)]
    mult = [all] * 5
    for layers in itertools.product(*mult):
        out1 = []
        for finalpos in generate_paths(layers, 4):
            out1.append(finalpos)
        out2 = []
        for finalpos in generate_paths(layers, 12):
            out2.append(finalpos)
        if len(out1) == 1 and len(out2) == 1:
            if out1[0] != out2[0]:
                print '-------------------'
                print out1, out2
                for l in layers:
                    print l


if __name__ == '__main__':
    test8()

    



