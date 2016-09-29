"""
"""
import numpy as np
import itertools
from getch import pause
import enum
import tables
import click
# import logging
# logging.basicConfig()
#
# log = logging.getLogger("")

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
    def __init__(self, row_id, pins, minimum=False):
        self.row_id = row_id
        self.filled = set([i for i in range(pins.size) if pins[i]])
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
    @property
    def all_rows(self):
        if not hasattr(self, '_rows'):
            self._rows = [r for r in self.generate_rows()]
        return self._rows


class SpacedRowFactory(RowFactory):
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
            count = 0
            # We simply use the binary encoding of an integer to generate the
            # combinations
            for i in xrange(self.size - 1, -1, -1):
                if (x >> i) & 1:
                    pins[i] = 1
                    count += 1

            # Then we filter those rows we deem valid
            if self.is_valid_layer(pins):
                yield Row(row_id, pins, count==self.minimum_pins)
                row_id += 1


class TopRowFactory(RowFactory):
    """Only generate pins around the slots where the ball is placed.
    
    This is for the top row, as it can help reduce combinations.
    """
    def __init__(self, size, slots):
        slots.sort()
        for i in slots:
            assert i > 0 and i < size
            # TODO: Should really check GAP too!

        self.size = size
        self.slots = slots

    def generate_rows(self):
        row_id = 0
        for pp in itertools.product(range(3), repeat=len(self.slots)):
            pins = np.zeros(self.size, np.int8)

            # Generate pins in all three relevant slots in front of the
            # hole...
            for pos, offset in zip(self.slots, pp):
                pins[pos - 1 + offset] = 1

            # assert self.is_valid_layer(pins)
            yield Row(row_id, pins)
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
                yield Box(self, layout)

    def construct_from_db(self, db_row):
        ids = db_row['rows']
        assert len(ids) == len(self.rows_of_variants)
        indexes = zip(range(len(ids)), ids)
        layout = [self.rows_of_variants[i][j] for (i, j) in indexes]
        return Box(self, layout)


class Box(object):
    def __init__(self, factory, layout):
        self.factory = factory
        self.layout = layout

    def _generate_paths(self, rows, pos, prob, cur=0):
        """A recursive generator that traces the path through each layer"""
        if cur == len(rows):
            yield pos, prob
        else:
            row = rows[cur]
            for newpos, newpr in row.mapping[pos]:
                for final_pos, final_pr in self._generate_paths(
                        rows, newpos, prob * newpr, cur+1):
                    yield final_pos, final_pr

    def get_distribution(self, positions):
        """Generate all distributions for every possible combination of rows"""
        buckets = np.zeros((len(positions), self.factory.output_count), np.double)

        for i, pos in enumerate(positions):
            for final_pos, final_pr in self._generate_paths(self.layout, pos, 1.0):
                buckets[i, final_pos] += final_pr

            buckets[i] /= buckets[i].sum()

        return buckets

    def dump(self):
        print '---'
        for r in self.layout:
            print r.pins
            print
        print '---'


class Database(object):
    def __init__(self, fname, factory, positions):
        self.factory = factory
        self.fname = fname
        self.positions = positions
        self.dtype = self.make_dtype(positions)

    def save_all(self):
        filters = tables.Filters(complib='blosc', complevel=5)
        h5 = tables.open_file(self.fname, 'w', filters=filters)

        attrs = h5.root._v_attrs
        attrs['positions'] = self.positions

        row = np.zeros(1, self.dtype)
        table = h5.create_table('/', 'output', self.dtype)
        cnt = 0
        for box in self.factory.generate_boxes():
            if cnt % 10000 == 0:
                click.echo('counting {}'.format(cnt))
            cnt += 1
            dist = box.get_distribution(self.positions)

            rids = [r.row_id for r in box.layout]
            row['rows'] = rids
            row['dist'] = dist
            table.append(row)
        h5.close()

    def read_all(self):
        h5 = tables.open_file(self.fname, 'r')
        return h5.root.output[:]

    def make_dtype(self, positions):
        return np.dtype([
            ('rows', int, len(self.factory.rows_of_variants)),
            ('dist', float, (len(positions), self.factory.output_count)),
        ])


def box_15_5_a_factory():
    # The basic box for the paper
    rf = SpacedRowFactory(15, False)
    rv = [rf.all_rows] * 5
    bk = BucketsRow(15, [
        (0, 1, 2, 3), 
        (3, 4, 5, 6, 7), 
        (7, 8, 9, 10, 11), 
        (11, 12, 13, 14),
    ])
    bf = BoxFactory(rv, bk)
    return bf

def box_15_5_b_factory():
    # The basic box for the paper
    rf = SpacedRowFactory(15, True)
    rv = [rf.all_rows] * 5
    bk = BucketsRow(15, [
        (0, 1, 2, 3), 
        (3, 4, 5, 6, 7), 
        (7, 8, 9, 10, 11), 
        (11, 12, 13, 14),
    ])
    bf = BoxFactory(rv, bk)
    return bf

def box_15_6_a_factory():
    # The basic box for the paper
    tf = TopRowFactory(15, [3, 11])
    rf = SpacedRowFactory(15, False)
    rv = [tf.all_rows]
    rv.extend([rf.all_rows] * 5)
    bk = BucketsRow(15, [
        (0, 1, 2, 3), 
        (3, 4, 5, 6, 7), 
        (7, 8, 9, 10, 11), 
        (11, 12, 13, 14),
    ])
    bf = BoxFactory(rv, bk)
    return bf


def box_9_4_a_factory():
    rf = SpacedRowFactory(9, False)
    rv = [rf.all_rows] * 4
    b = BucketsRow(9, [
        (0, 1, 2), 
        (3, 4, 5), 
        (6, 7, 8),
    ])
    return BoxFactory(rv, b)


# ---------------------------------------------------------------------------
# Commands below here
#
@click.group()
def waddington():
    pass

@waddington.command()
def save_box_15_5_a():
    click.echo('Creating factory')
    factory = box_15_5_a_factory()
    db = Database('15_5_a.h5', factory, [3, 11])
    db.save_all()


@waddington.command()
def save_box_15_5_b():
    click.echo('Creating factory')
    factory = box_15_5_b_factory()
    db = Database('15_5_b.h5', factory, [3, 11])
    db.save_all()


@waddington.command()
def save_box_15_6_a():
    click.echo('Creating factory')
    factory = box_15_6_a_factory()
    db = Database('15_6_a.h5', factory, [3, 11])
    db.save_all()


@waddington.command()
@click.argument('rows', default=4)
@click.argument('pins', default=15)
@click.option('--few', is_flag=True)
# @click.option('--top', type=int, default=0)
def quantify(rows, pins, few, top):
    # if top:
    #     top = TopRowFactory(pins, 
    rf = SpacedRowFactory(pins, few)
    print len(rf.all_rows) ** rows


@waddington.command()
def test_box_15_5_a():
    factory = box_15_5_a_factory()
    db = Database('15_5_a.h5', factory, [3, 11])
    data = db.read_all()
    print len(data)
    dist = data['dist']
    tups = set([tuple(d.ravel()) for d in dist])
    print len(tups)
    # print shorter
    # print shorter[:,1,2:]
    # print shorter[:,0,:2]
    # short
    # match = [[0, 0, 0, 0], [0, 0, 0, 0]]
    # matching = (shorter == match)
    # m = matching.sum(axis=2).sum(axis=1)
    # ind = np.where(m==8)[0]
    # print ind
    # print shorter[ind[0]]
    # print factory.construct_from_db(data[ind[0]]).dump()
    
@waddington.command()
def test():
    tf = TopRowFactory(15, [3, 11])
    for row in tf.generate_rows():
        print row.pins



if __name__ == '__main__':
    waddington()
#     save_box_15_5_b()
#     save_box_15_5_a()
    # test_box_15_5_a()



