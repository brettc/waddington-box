"""
"""
import numpy as np
import itertools
# from getch import pause
import enum
import tables

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
    def __init__(self, row_id, pins):
        self.row_id = row_id
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

        def legal_move(p):
            return p > 0 and p < self.pins.size - 1

        if hit == Hit.middle:
            # Only go in that direction if it possible
            newpos = pos - 2
            if legal_move(newpos):
                yield newpos

            newpos = pos + 2
            if legal_move(newpos):
                yield newpos

        elif hit == Hit.left:
            newpos = pos + 1
            if legal_move(newpos):
                yield newpos
            else:
                assert legal_move(pos - 3)
                yield pos - 3

        elif hit == Hit.right:
            newpos = pos - 1
            if legal_move(newpos):
                yield newpos
            else:
                assert legal_move(pos + 3)
                yield pos + 3

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
        """A dictionary that describes all of the possible i/o mappings
        
        A row maps inputs at pin positions to possibly multiple pin output
        positions.
        """
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
        row_id = 0
        for x in range(2 ** self.size):
            pins = np.zeros(self.size, np.int8)
            # We simply use the binary encoding of an integer to generate the
            # combinations
            for i in xrange(self.size -1, -1, -1):
                pins[i] = (x >> i) & 1 

            if self.is_valid_layer(pins):
                yield Row(row_id, pins)
                row_id += 1

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
        row_id = 0
        for pp in itertools.product(range(4), repeat=len(self.positions)):
            pins = np.zeros(self.size, np.int8)

            for pos, offset in zip(self.positions, pp):
                if offset != 0:
                    pins[pos - 2 + offset] = 1

            # assert self.is_valid_layer(pins)
            yield Row(pins)
            row_id += 1

    @property
    def all_rows(self):
        if not hasattr(self, '_rows'):
            self._rows = [r for r in self.generate_rows()]
        return self._rows



class Database(object):
    def __init__(self):
        filters = tables.Filters(complib='blosc', complevel=5)
        h5 = tables.open_file(str(self.path), 'w', filters=filters)
        

    def _dtype(self):
        return np.dtype([
            ('generation', int),
            ('target', int),
            ('best', float),
            ('indexes', int, self._size),
        ])


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


def recurse_rows(rows_of_rows, input_output, box, row_num, row_max, pin_count):
    if row_num == row_max:
        yield box, input_output
        return
        # # print '---'
        # # for row in box:
        # #     print row.pins
        # return 

    rows = rows_of_rows[row_num]
    inp = input_output[row_num]
    out = input_output[row_num + 1]

    for row in rows:
        box[row_num] = row
        # if row_num + 1 < row_max:
        out[:] = 0

        # Go through all the inputs and get the outputs. Only look at possible
        # positions (not edges)
        pos = 1
        while pos < pin_count - 1:
            qty = inp[pos]
            if qty:
                for opos in row.mapping[pos]:
                    out[opos] += qty
            pos += 1

        for b, io in recurse_rows(rows_of_rows, input_output, box, row_num + 1, row_max, pin_count):
            yield b, io



# Breadth first
def generate_boxes(rows_of_rows, pos, pin_count):
    total_rows = len(rows_of_rows)
    box = [None] * total_rows
    input_output = np.zeros((total_rows + 1, pin_count), dtype=int)
    input_output[0, pos] = 1
    for b, io in recurse_rows(rows_of_rows, input_output, box, 0, total_rows, pin_count):
        yield b, io
    print input_output[-1]


def test_recurse():
    ff = RowFactory(9)
    rows_of_rows = [ff.all_rows] * 4
    print("Num: {}".format(len(ff.all_rows) ** 4))
    dist = np.zeros(9, int)
    for b, io in generate_boxes(rows_of_rows, 4, 9):
        dist += io[-1]
    print dist


def generate_paths(rows, pos, cur=0):
    if cur == len(rows):
        yield pos
    else:
        row = rows[cur]
        for newpos in row.mapping[pos]:
            for finalpos in generate_paths(rows, newpos, cur+1):
                yield finalpos


def test_paths():
    ff = RowFactory(9)
    dist = np.zeros(9, dtype=int)
    rows_of_rows = [ff.all_rows] * 4
    for rows in itertools.product(*rows_of_rows):
        # outputs = []
        for finalpos in generate_paths(rows, 4):
            dist[finalpos] += 1
    print dist


if __name__ == '__main__':
    import timeit

    # Timing
    print timeit.Timer(stmt="test_recurse()",
                       setup="from __main__ import test_recurse",
                       ).timeit(1)
    print timeit.Timer(stmt="test_paths()",
                       setup="from __main__ import test_paths",
                       ).timeit(1)
    # test_recurse()
    # test_paths()

    



