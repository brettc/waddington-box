"""
"""
import numpy as np
import itertools
import os
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
    def __init__(self, size, slots, empty=True):
        slots.sort()
        for i in slots:
            assert i > 0 and i < size
            # TODO: Should really check GAP too!

        self.size = size
        self.slots = slots
        self.empty = empty

    def generate_rows(self):
        row_id = 0
        pins_per_slot = 4 if self.empty else 3
        for pp in itertools.product(range(pins_per_slot), repeat=len(self.slots)):
            pins = np.zeros(self.size, np.int8)

            # Generate pins in all three relevant slots in front of the
            # hole...
            for pos, offset in zip(self.slots, pp):
                if offset != 3:
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
        self.possible_rows = rv
        self.buckets = buckets

        # If we supply a bucketing row, then append it
        if self.buckets:
            self.output_count = self.buckets.bucket_count
            self.possible_rows.append([self.buckets])
        else:
            self.output_count = rv[-1][0].pins.size

        self.dimensions = [len(rs) for rs in self.possible_rows]

    def calc_maximum_boxes(self):
        # Note: doesn't account for row filtering

        # Don't count bucketing row
        if self.buckets:
            rv = self.possible_rows[:-1]
        else:
            rv = self.possible_rows

        tot = 1
        for r in rv:
            tot *= len(r)

        return tot

    def generate_boxes(self):
        box_num = 0
        for layout in itertools.product(*self.possible_rows):
            row_iter = iter(layout)
            row_now = row_iter.next()
            for row_next in row_iter:
                if not row_now.can_be_consecutive(row_next):
                    break

                row_now = row_next
            else:
                # If we made it all the way through, then its okay!
                yield Box(self, box_num, layout)
            box_num += 1

    def from_ident(self, ident):
        # Convert index into the dimensions in the rows
        ids = np.unravel_index(ident, self.dimensions)
        assert len(ids) == len(self.possible_rows)
        indexes = zip(range(len(ids)), ids)
        layout = [self.possible_rows[i][j] for (i, j) in indexes]
        return Box(self, ident, layout)


class Box(object):
    def __init__(self, factory, ident, layout):
        self.factory = factory
        self.ident = ident
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

class FileExistsError(click.ClickException):
    pass

class Database(object):
    def __init__(self, fname, factory=None, positions=None,
                 overwrite=False):
        if os.path.exists(fname):
            if factory is not None and not overwrite:
                raise FileExistsError("File {} already exists!".format(fname))

        self.fname = fname
        if factory is None:
            self.load()
        else:
            self.factory = factory
            self.positions = positions

        self.dtype = self.make_dtype()

    def load(self):
        h5 = tables.open_file(self.fname, 'r')
        attrs = h5.root._v_attrs
        self.factory = attrs.factory
        self.positions = attrs.positions
        self.data = self.read_all()
        self.dists = self.data['dist']
        self.ids = self.data['ident']
        h5.close()

    def _resave_old(self):
        filters = tables.Filters(complib='blosc', complevel=5)
        h5 = tables.open_file(self.fname, 'r+', filters=filters)
        attrs = h5.root._v_attrs
        attrs['positions'] = self.positions
        attrs['factory'] = self.factory
        
    def save_all(self):
        filters = tables.Filters(complib='blosc', complevel=5)
        h5 = tables.open_file(self.fname, 'w', filters=filters)

        attrs = h5.root._v_attrs
        attrs['positions'] = self.positions
        attrs['factory'] = self.factory

        row = np.zeros(1, self.dtype)
        table = h5.create_table('/', 'output', self.dtype)

        last_box = 0
        final_box = self.factory.calc_maximum_boxes()
        with click.progressbar(label="Boxing", length=final_box) as bar:
            for box in self.factory.generate_boxes():
                dist = box.get_distribution(self.positions)
                row['ident'] = box.ident
                row['dist'] = dist
                table.append(row)

                boxes_done = box.ident - last_box
                if boxes_done > 10000:
                    bar.update(boxes_done)
                    last_box = box.ident

            bar.update(final_box)

        h5.close()

    def read_all(self):
        h5 = tables.open_file(self.fname, 'r')
        return h5.root.output[:]

    def make_dtype(self):
        return np.dtype([
            ('ident', int),
            ('dist', float, (len(self.positions), self.factory.output_count)),
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

def box_15_5_c_factory():
    # The basic box for the paper
    tf = TopRowFactory(15, [3, 11])
    rf = SpacedRowFactory(15, False)
    rv = [tf.all_rows]
    rv.extend([rf.all_rows] * 4)
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
def save_box_9_4_a():
    click.echo('Creating factory')
    factory = box_9_4_a_factory()
    db = Database('9_4_a.h5', factory, [4])
    db.save_all()

@waddington.command()
def save_box_15_5_a():
    click.echo('Creating factory...')
    factory = box_15_5_a_factory()
    click.echo('Maximum Boxes = {}'.format(factory.calc_maximum_boxes()))
    # db = Database('15_5_a.h5', factory, [3, 11])
    # db.save_all()


@waddington.command()
def save_box_15_5_b():
    click.echo('Creating factory')
    factory = box_15_5_b_factory()
    db = Database('15_5_b.h5', factory, [3, 11])
    db.save_all()

@waddington.command()
def save_box_15_5_c(overwrite):
    click.echo('Creating factory...')
    factory = box_15_5_c_factory()
    click.echo('Maximum Boxes = {}'.format(factory.calc_maximum_boxes()))
    db = Database('15_5_c.h5', factory, [3, 11], overwrite)
    db.save_all()

@waddington.command()
def save_box_15_6_a():
    click.echo('Creating factory')
    factory = box_15_6_a_factory()
    db = Database('15_6_a.h5', factory, [3, 11])
    db.save_all()

@waddington.command()
def save_box_main():
    click.echo('Creating factory...')
    factory = box_15_5_c_factory()
    click.echo('Maximum Boxes = {}'.format(factory.calc_maximum_boxes()))
    db = Database('main.h5', factory, [3, 11])
    db.save_all()


@waddington.command()
@click.argument('rows', default=4)
@click.argument('pins', default=15)
@click.option('--few', is_flag=True)
# @click.option('--top', type=int, default=0)
def quantify(rows, pins, few):
    # if top:
    #     top = TopRowFactory(pins, 
    rf = SpacedRowFactory(pins, few)
    print len(rf.all_rows) ** rows


@waddington.command()
def test_2():
    print 'here'
    # factory = box_15_5_a_factory()
    db = Database('15_5_a.h5')
    f = db.factory
    print db.distributions
    print f
    # db._resave_old()
    # data = db.read_all()
    # print len(data)
    # dist = data['dist']
    # tups = set([tuple(d.ravel()) for d in dist])
    # print len(tups)
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
def analysis():
    db = Database('main.h5')
    print db.factory
    print len(db.dists)

    # with click.progressbar(db.data) as boxes:
    #     for b in boxes:
    for b in db.data:
        d = b['dist']
        ident = b['ident']
        if d[0][0] == 1.0:
            if d[1][2] == 1.0:
                print d
                box = db.factory.from_ident(ident)
                box.dump()
        # if d[0][2] == 1.0:
        #     if d[1][2] == 1.0:
        #         print d
        #         box = db.factory.construct_from_db(b)
        #         box.dump()
        # if d[0][0] == 0.75:
        #     if d[0][1] == 0.25:
        #         if d[1][2] == 0.25:
        #             if d[1][3] == 0.75:
        #                 print d
        #                 box = db.factory.construct_from_db(b)
        #                 box.dump()
            
@waddington.command()
def test():
    db = Database('main.h5')
    # tups = dict([tuple(d.ravel()) for d in db.dists))
    # dims = [len(rs) for rs in f.possible_rows]
    # for b in f.generate_boxes():
    #     print '--'
    #     print np.unravel_index(b.ident, dims)
    #     print [r.row_id for r in b.layout]
    #



    # tf = TopRowFactory(15, [3, 11], empty=False)
    # for row in tf.generate_rows():
    #     print row.pins

    # rv = [rf.all_rows] * 5
    # f = BoxFactory(rv)
    # db = Database('test.h5', f, [5])
    # db.save_all()

    # print f.calc_maximum_boxes()
    # for box in f.generate_boxes():
    #     pass
    # print box.ident



if __name__ == '__main__':
    waddington()
    # test()
    # test_box_15_5_a()
    # save_box_15_5_b()
    # save_box_15_5_a()



