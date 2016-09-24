"""
"""
import numpy as np
import itertools
from getch import pause
import enum

class Hit(enum.Enum):
    open = 0
    middle = 1
    left = 2
    right = 3

HITS = {
    (0, 0, 0) : Hit.open,
    (0, 1, 0) : Hit.middle,
    (1, 0, 0) : Hit.left,
    (0, 0, 1) : Hit.right,
}

class Row(object):
    def __init__(self, pins):
        self.pins = pins

    def hit_test(self, pos):
        # We assume our ball has width, so the position cannot be right on the
        # edges!
        assert self.pins.shape[0] == self.pins.size
        if pos < 1 or pos >= self.pins.size - 1:
            raise ValueError("Bad Position")

        # Get the what's in front of the ball
        block = tuple(self.pins[pos - 1: pos + 2])
        try:
            hit = HITS[block]
        except:
            raise ValueError("Bad layer: more than one pin")

        return hit

    def generate_moves(self, pos):
        hit = self.hit_test(pos)

        if hit == Hit.middle:
            # Only go in that direction if it possible
            if pos - 2 > 0:
                yield pos - 2
            if pos + 2 < self.pins.size - 2:
                yield pos + 2

        elif hit == Hit.left:
            if pos + 1 < self.pins.size - 2 :
                yield pos + 1

        elif hit == Hit.right:
            if pos - 1 > 0:
                yield pos - 1

        else:
            yield pos

    @property
    def mapping(self):
        if not hasattr(self, '_mapping'):
            self._mapping, self._output = self.generate_mapping()
        return self._mapping

    @property
    def output(self):
        if not hasattr(self, '_output'):
            self._mapping, self._output = self.generate_mapping()
        return self._output

    def generate_mapping(self, positions=[]):
        if not positions:
            positions = range(1, self.pins.size-1)
        mapping = {}
        dist = {}
        for in_p in positions:
            out = [out_p for out_p in self.generate_moves(in_p)]

            for o in out:
                try:
                    dist[o] += 1
                except KeyError:
                    dist[o] = 1

            mapping[in_p] = out
        return mapping, tuple(dist.items())

    def show_moves(self, pos):
        mapped = [p for p in self.generate_moves(pos)]
        top = np.zeros_like(self.pins)
        top[pos-1:pos+2] = 8
        bot = np.zeros_like(self.pins)
        for p in mapped:
            bot[p-1:p+2] = 8
        print('IDX: {}'.format(np.arange(self.pins.size)))
        print('IN : {}'.format(top))
        print('ROW: {}'.format(self.pins))
        print('OUT: {}'.format(bot))


class RowFactory(object):
    def __init__(self, size):
        self.size = size

    def is_valid_layer(self, pins):
        # Test 1: at least size/4 pins
        pin_count = pins.sum()
        if pin_count < self.size // 4:
            return False

        # Test 2: pins stand alone, and have gaps of 3
        for i in range(self.size-3):
            window = pins[i:i+4]
            if window.sum() > 1:
                return False

        return True

    def generate_rows(self):
        # There is a pin every 3 holes
        for x in range(2 ** self.size):
            pins = np.zeros(self.size, np.int8)
            # We simply use the binary encoding of an integer to generate the
            # combinations
            for i in xrange(self.size -1, -1, -1):
                pins[i] = (x >> i) & 1 

            if self.is_valid_layer(pins):
                yield Row(pins)

    @property
    def all_rows(self):
        if not hasattr(self, '_rows'):
            self._rows = [r for r in self.generate_rows()]
        return self._rows


class ConstrainedRowFactory(object):
    """Only generate pins around the assigned positions"""
    def __init__(self, size, positions):
        positions.sort()
        for i in positions:
            assert i > 0 and i < size
            # Should check GAP too!

        self.size = size
        self.positions = positions

    def generate_rows(self):
        for pp in itertools.product(range(4), repeat=len(self.positions)):
            pins = np.zeros(self.size, np.int8)

            for pos, offset in zip(self.positions, pp):
                if offset != 0:
                    pins[pos - 2 + offset] = 1

            # assert self.is_valid_layer(pins)
            yield Row(pins)

    @property
    def all_rows(self):
        if not hasattr(self, '_rows'):
            self._rows = [r for r in self.generate_rows()]
        return self._rows


class Mappings(object):
    def __init__(self, fact):
        assert isinstance(fact, RowFactory)
        lk = {}
        for row in fact.generate_rows():
            m = lk.setdefault(row.output, [])
            m.append(row)

        self.lookup = lk


def test_moves():
    # row = Row(np.asarray([0, 0, 0, 1, 0, 0, 0, 1], np.int8))
    # row.show_moves(3)
    # print(row.generate_mapping())
    #
    # rf = RowFactory(9)
    # for row in rf.generate_rows():
    #     print row.pins
    #
    #
    # mm = Mappings(RowFactory(17))
    # print len(mm.lookup)
    # for k, v in mm.lookup.iteritems():
    #     if len(v) > 1:
    #         print k, len(v)
    #
    ff = RowFactory(17)
    cf = ConstrainedRowFactory(17, [4, 12])
    ffl = len(ff.all_rows)
    cfl = len(cf.all_rows)
    # for row in cf.generate_rows():
    #     print row.pins
    print cfl * (ffl ** 3)

def recurse_rows(rows_of_rows, input_output, box, row_num, row_max):
    if row_num == row_max:
        # print '---'
        # for row in box:
        #     print row.pins
        return 

    rows = rows_of_rows[row_num]

    for row in rows:
        box[row_num] = row
        if row_num + 1 < row_max:
            input_output[row_num + 1] = 0

        # Go through all the inputs and get the outputs
        for pos, qty in enumerate(input_output[row_num]):
            if qty:
                for out in row.mapping[pos]:
                    input_output[row_num + 1, out] += qty

        recurse_rows(rows_of_rows, input_output, box, row_num + 1, row_max)


def generate_boxes(rows_of_rows, pos, pin_count):
    total_rows = len(rows_of_rows)
    box = [None] * total_rows
    input_output = np.zeros((total_rows + 1, pin_count), dtype=int)
    input_output[0, pos] = 1
    recurse_rows(rows_of_rows, input_output, box, 0, total_rows)
    print input_output[-1]


def test_recurse():
    ff = RowFactory(11)
    rows_of_rows = [ff.all_rows] * 2
    generate_boxes(rows_of_rows, 5, 11)







if __name__ == '__main__':
    # test_moves()
    test_recurse()

    



