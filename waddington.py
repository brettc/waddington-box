"""
"""
import numpy as np
import itertools
from getch import pause
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
    def __init__(self, row_id, pins, filled, minimum=False):
        self.row_id = row_id
        self.filled = filled
        self.minimum = minimum
        self.pins = np.asarray(pins, dtype=np.int8)
        m = {}
        for p in range(self.pins.size):
            m[p] = self.moves(p)
        self.mapping = m

    def can_be_consecutive(self, other_row):
        # It might be Buckets row...
        if not isinstance(other_row, Row):
            return True

        # We don't want pins below other pins
        if self.filled & other_row.filled:
            return False

        # Don't have two rows with few pins 
        if self.minimum and other_row.minimum:
            return False

        return True

    def hit_test(self, pos):
        # position must be line up with pins
        assert self.pins.shape[0] == self.pins.size
        if pos < 0 or pos >= self.pins.size:
            raise ValueError("Bad Position")

        # Get the what's in front of the ball, adjusting for edge conditions.
        block = (
            self.pins[pos - 1] if pos > 0 else 0,
            self.pins[pos],
            self.pins[pos + 1] if pos < self.pins.size - 1 else 0
        )

        try:
            hit = HITS[block]
        except:
            raise ValueError("Bad layer: more than one pin")

        return hit

    def moves(self, pos):
        hit = self.hit_test(pos)
        moves = []

        def legal_move(p):
            return p >= 0 and p < self.pins.size

        if hit == Hit.middle:
            # If we're hit in the middle, we have to move 2 spaces to go
            # around the pin. Either way is possible (equal chance).
            left_ok = legal_move(pos - 2)
            rght_ok = legal_move(pos + 2)

            if left_ok and rght_ok:
                moves.append((pos - 2, .5))
                moves.append((pos + 2, .5))
            elif left_ok:
                moves.append((pos - 2, 1.0))
            else:
                moves.append((pos + 2, 1.0))

        elif hit == Hit.left:
            # Hit on the left, we just need to move one space. It is possible
            # that we're on the edge though -- then we "bounce" and go the
            # other way.
            newpos = pos + 1
            if legal_move(newpos):
                moves.append((newpos, 1.0))
            else:
                assert legal_move(pos - 3)
                moves.append((pos - 3, 1.0))

        elif hit == Hit.right:
            # Opposite of above.
            newpos = pos - 1
            if legal_move(newpos):
                moves.append((newpos, 1.0))
            else:
                assert legal_move(pos + 3)
                moves.append((pos + 3, 1.0))

        else:
            # Fall through
            moves.append((pos, 1.0))

        return tuple(moves)


def test_rows():
    # Left Edge
    row = Row(0, [1, 0, 0, 0, 0])
    assert row.hit_test(0) == Hit.middle
    assert row.mapping[0] == ((2, 1.0),)

    assert row.hit_test(1) == Hit.left
    assert row.mapping[0] == ((2, 1.0),)

    # Left Edge 2
    row = Row(0, [0, 1, 0, 0, 0])
    assert row.hit_test(0) == Hit.right
    assert row.mapping[0] == ((3, 1.0),)

    assert row.hit_test(1) == Hit.middle
    assert row.mapping[1] == ((3, 1.0),)

    assert row.hit_test(2) == Hit.left
    assert row.mapping[2] == ((3, 1.0),)

    # Left Edge 3
    row = Row(0, [0, 0, 1, 0, 0])
    assert row.hit_test(1) == Hit.right
    assert row.mapping[1] == ((0, 1.0),)

    # Right edge
    row = Row(0, [0, 0, 0, 1])
    assert row.hit_test(2) == Hit.right
    assert row.mapping[2] == ((1, 1.0),)

    assert row.hit_test(3) == Hit.middle
    assert row.mapping[3] == ((1, 1.0),)

    # Right edge 2
    row = Row(0, [0, 0, 0, 1, 0])
    assert row.hit_test(2) == Hit.right
    assert row.mapping[2] == ((1, 1.0),)

    assert row.hit_test(3) == Hit.middle
    assert row.mapping[3] == ((1, 1.0),)

    assert row.hit_test(4) == Hit.left
    assert row.mapping[4] == ((1, 1.0),)

    # Right Edge 3
    row = Row(0, [0, 0, 1, 0, 0])
    assert row.hit_test(3) == Hit.left
    assert row.mapping[3] == ((4, 1.0),)

    # Both sides
    row = Row(0, [0, 0, 0, 1, 0, 0, 0])
    assert row.hit_test(2) == Hit.right
    assert row.mapping[2] == ((1, 1.0),)

    assert row.hit_test(3) == Hit.middle
    assert row.mapping[3] == ((1, 0.5),(5, 0.5))

    assert row.hit_test(4) == Hit.left
    assert row.mapping[4] == ((5, 1.0),)


class RowFactory(object):
    def __init__(self, size, few=True):
        self.size = size
        self.minimum_pins = self.size // 4
        if few == True:
            self.minimum_pins -= 1

    def is_valid_layer(self, pins):
        # Test 1: at least size/4 pins
        pin_count = pins.sum()
        if pin_count < self.minimum_pins:
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
            filled = set()
            count = 0
            # We simply use the binary encoding of an integer to generate the
            # combinations
            for i in xrange(self.size - 1, -1, -1):
                if (x >> i) & 1:
                    pins[i] = 1
                    filled.add(i)
                    count += 1

            # Then we filter those rows we deem valid
            if self.is_valid_layer(pins):
                yield Row(row_id, pins, filled, count==self.minimum_pins)
                row_id += 1

    @property
    def all_rows(self):
        if not hasattr(self, '_rows'):
            self._rows = [r for r in self.generate_rows()]
        return self._rows


class ConstrainedRowFactory(object):
    """Only generate pins around the assigned positions
    
    This is for the top row, as it can help reduce combinations.
    """
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


class BucketsRow(object):
    """Generate the mapping for the final row that buckets the balls"""
    def __init__(self, pin_count, groups):
        self.row_id = 0
        self.pin_count = pin_count
        self.bucket_count = len(groups)

        # Some sanity checking. All pins must be there.
        all_pins = set()
        for grp in groups:
            for p in grp:
                all_pins.add(p)

        assert set(range(pin_count)) == all_pins

        # Ok, now generate the bucket mapping
        map_a = {}
        for g_i, grp in enumerate(groups):
            for p in grp:
                outs = map_a.setdefault(p, [])
                outs.append(g_i)

        # Right, now go through and set the probabilities. We need this as
        # some pin positions lie between the buckets.
        map_b = {}
        for p_in, p_outs in map_a.items():
            prob = 1.0 / len(p_outs)
            map_b[p_in] = tuple([(p, prob) for p in p_outs])

        self.mapping = map_b

        # TODO: This is rubbish. Make it print something clever.
        self.pins = groups
        

class BoxFactory(object):
    def __init__(self, rv, buckets=None):
        self.rows_of_variants = rv
        self.buckets = buckets

        # If we supply a bucketing row, then append it
        if self.buckets:
            self.output_count = self.buckets.bucket_count
            self.rows_of_variants.append([self.buckets])
        else:
            self.output_count = rv[-1][0].pins.size

    def generate_boxes(self):
        for layout in itertools.product(*self.rows_of_variants):
            row_iter = iter(layout)
            row_now = row_iter.next()
            for row_next in row_iter:
                if not row_now.can_be_consecutive(row_next):
                    break
                row_now = row_next
            else:
                # If we made it all the way through, then its okay!
                yield layout

    def generate_paths(self, rows, pos, prob, cur=0):
        """A recursive generator that traces the path through each layer"""
        if cur == len(rows):
            yield pos, prob
        else:
            row = rows[cur]
            for newpos, newpr in row.mapping[pos]:
                for final_pos, final_pr in self.generate_paths(
                        rows, newpos, prob * newpr, cur+1):
                    yield final_pos, final_pr

    def generate_distributions(self, positions):
        """Generate all distributions for every possible combination of rows"""
        buckets = np.zeros((len(positions), self.output_count), np.double)

        for layout in self.generate_boxes():
            # layout contains one combination of possible rows. Now we just
            # trace the paths through it.
            buckets[:, :] = 0.0
            for i, pos in enumerate(positions):
                for final_pos, final_pr in self.generate_paths(layout, pos, 1.0):
                    buckets[i, final_pos] += final_pr

                buckets[i] /= buckets[i].sum()

            yield layout, buckets

    def make_dtype(self, positions):
        return np.dtype([
            ('rows', int, len(self.rows_of_variants)),
            ('dists', float, (len(positions), self.output_count)),
        ])

    def show(self, ids):
        print ids
        indexes = zip(range(len(ids)), ids)
        rows = [self.rows_of_variants[i][j] for (i, j) in indexes]
        for r in rows:
            print r.pins


def test_generate_boxes():
    rf = RowFactory(15, False)
    rv = [rf.all_rows] * 4
    bf = BoxFactory(rv)
    all = [l for l in bf.generate_boxes()]
    print len(all)
    for a in all:
        for r in a:
            print r.pins
        pause()


def test_waddington_box():
    print
    rf = RowFactory(11, False)
    rv = [rf.all_rows] * 4
    b = BucketsRow(11, [
        (0, 1, 2, 3), 
        (4, 5, 6), 
        (7, 8, 9, 10),
    ])
    wb = BoxFactory(rv, b)
    # map = {}
    for layout, buckets in wb.generate_distributions([0, 10]):
        if buckets[0, 1] == 0.25 and buckets[1, 1] == 1.0:
            for b in buckets:
                print b
            for r in layout:
                print r.pins
            break

        # dist = tuple(buckets.ravel())
        # all = map.setdefault(dist, [])
        # all.append(layout)

    # print len(map)

    # v = [len(v) for v in map.values()]
    # m = min(v)
    # print 'min', m
    # for k, v in map.items():
    #     if len(v) == m:
    #         print '----'
    #         print k
            # for l in v:
            #     print
            #     for r in l:
            #         print r.pins
            #     print '-'


class WaddingtonBox_4x9(BoxFactory):
    def __init__(self):
        super(WaddingtonBox_4x9, self).__init__()
        rf = RowFactory(9)
        br = BucketsRow(9, [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
        self.bucket_count = br.bucket_count
        self.rows_of_variants.extend([rf.all_rows] * 4)
        self.rows_of_variants.append([br])

    def write_distributions(self, fname, positions):
        filters = tables.Filters(complib='blosc', complevel=5)
        self.h5 = tables.open_file(fname, 'w', filters=filters)
        dtype = self.make_dtype(positions)
        tab = self.h5.create_table('/', 'output', dtype)
        attrs = self.h5.root._v_attrs
        attrs['positions'] = positions
        row = np.zeros(1, dtype)
        for rows, buckets in self.generate_distributions(positions):
            rids = [r.row_id for r in rows]
            row['rows'] = rids
            row['dists'] = buckets
            tab.append(row)


def test_4x9():
    wb = WaddingtonBox_4x9()
    # for rows, buckets in wb.generate_distributions(4):
    #     pass
    #
    # wb.write_distributions('test.h5', [2, 4])
    #
    h5 = tables.open_file('test.h5', 'r')
    k = h5.root.output[:]
    print len(k)
    print k[99]
    print len(wb.rows_of_variants)
    print [len(v) for v in wb.rows_of_variants]

    # print np.unravel_index(99, lens)
    wb.show(k['rows'][99])


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
    ff = RowFactory(17)
    rows_of_rows = [ff.all_rows] * 5
    maxn = len(ff.all_rows) ** 5
    print("Num: {}".format(len(ff.all_rows) ** 5))
    dist = np.zeros(9, int)
    i = 0
    for b, io in generate_boxes(rows_of_rows, 8, 17):
        if i % 100000 == 0:
            print i, maxn
        i += 1
        pass
        # dist += io[-1]
    print dist


def timings():
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


if __name__ == '__main__':
    test_generate_boxes()
    # test_recurse()
    # test_4x9()



