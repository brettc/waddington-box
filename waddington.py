from __future__ import print_function
import numpy as np
import itertools
import os
import enum
import tables
import click
from tabulate import tabulate


FINAL_FILENAME = 'main.h5'


class Hit(enum.Enum):
    open = 0
    middle = 1
    left = 2
    right = 3

HITS = {
    (0, 0, 0): Hit.open,
    (0, 1, 0): Hit.middle,
    (1, 0, 0): Hit.left,
    (0, 0, 1): Hit.right,
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
    assert row.mapping[3] == ((1, 0.5), (5, 0.5))

    assert row.hit_test(4) == Hit.left
    assert row.mapping[4] == ((5, 1.0),)


class RowFactory(object):

    @property
    def all_rows(self):
        if not hasattr(self, '_rows'):
            self._rows = [r for r in self.generate_rows()]
        return self._rows


class SpacedRowFactory(RowFactory):

    def __init__(self, size):
        self.size = size
        self.minimum_pins = self.size // 4

    def is_valid_layer(self, pins):
        # Test 1: at least size/4 pins
        pin_count = pins.sum()
        if pin_count < self.minimum_pins:
            return False

        # Test 2: pins stand alone, and have gaps of 3
        for i in range(self.size - 3):
            window = pins[i:i + 4]
            if window.sum() > 1:
                return False

        return True

    def generate_rows(self):
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
                yield Row(row_id, pins, count == self.minimum_pins)
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

        # TODO: This is rubbish. But it allows us to print the pins on every
        # row
        self.pins = groups

    def overlapping(self):
        boundaries = []
        for inp, outputs in self.mapping.items():
            if len(outputs) > 1:
                boundaries.append(inp)
        return boundaries


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

    def from_ident(self, ident, index=None):
        # Convert index into the dimensions in the rows
        ids = np.unravel_index(ident, self.dimensions)
        assert len(ids) == len(self.possible_rows)
        indexes = zip(range(len(ids)), ids)
        layout = [self.possible_rows[i][j] for (i, j) in indexes]
        return Box(self, ident, layout, index=index)


class Box(object):

    def __init__(self, factory, ident, layout, index=None):
        self.factory = factory
        self.ident = ident
        self.layout = layout
        self.index = index

    def total_pins(self):
        tot = 0
        for r in self.layout:
            if not isinstance(r, BucketsRow):
                tot += r.pins.sum()
        return tot

    def _generate_paths(self, rows, pos, prob, cur=0):
        """A recursive generator that traces the path through each layer"""
        if cur == len(rows):
            yield pos, prob
        else:
            row = rows[cur]
            for newpos, newpr in row.mapping[pos]:
                for final_pos, final_pr in self._generate_paths(
                        rows, newpos, prob * newpr, cur + 1):
                    yield final_pos, final_pr

    def get_distribution(self, positions):
        """Generate all distributions for every possible combination of rows"""
        buckets = np.zeros(
            (len(positions), self.factory.output_count), np.double)

        for i, pos in enumerate(positions):
            for final_pos, final_pr in self._generate_paths(self.layout, pos, 1.0):
                buckets[i, final_pos] += final_pr

            buckets[i] /= buckets[i].sum()

        return buckets

    def dump(self, positions=None):
        """Print a picture of it to the console"""
        if self.factory.buckets:
            rows = self.layout[:-1]
        else:
            rows = self.layout

        beg = "| "
        end = " |"

        r1 = rows[0].pins
        width = len(r1) * 3 + len(beg) + len(end) + 2
        print("=" * width)

        if self.index:
            print("Index {}".format(self.index).center(width))
        else:
            print("Ident {}".format(self.ident).center(width))

        if positions:
            keys = np.zeros(len(r1), int)
            keys[positions] = 1
            text = "".join([('---', '   ')[b] for b in keys])
            print(beg, text, end)
        else:
            print(beg, '---' * len(r1), end)

        for r in rows:
            text = "".join([(' - ', '(0)')[p] for p in r.pins])
            print(beg, "   " * len(r.pins), end)
            print(beg, text, end)
            print(beg, "   " * len(r.pins), end)

        if self.factory.buckets:
            buck_row = self.layout[-1]
            bound = buck_row.overlapping()
            keys = np.zeros(buck_row.pin_count, int)
            keys[bound] = 1
            text = "".join([('   ', ' | ')[b] for b in keys])
            print(beg, text, end)

        print(beg, '---' * len(r.pins), end)

        if positions:
            dists = self.get_distribution(positions)
            rows = [['Slot {}'.format(i + 1)] + list(vals)
                    for (i, vals) in enumerate(dists)]
            headers = ["Buck {}".format(i + 1) for i in range(dists.shape[1])]
            print(tabulate(rows, headers=headers))

            # Now print entropies / MI
            pr = dists / 2.0
            ent_p_b = calc_entropy(pr)
            ent_b = round(calc_entropy(pr.sum(axis=0)), 3)
            mi = round(1.0 + ent_b - ent_p_b, 3)
            print()
            print("H(S) = 1.0 / H(B) = {} / I(S; B) = {}".format(ent_b, mi).center(width))

        print("=" * width)

    def dump_tikz(self, positions=None):
        if self.factory.buckets:
            ll = self.layout[:-1]
        else:
            ll = self.layout[:]

        print("index: {}".format(self.index))
        print("ident: {}".format(self.ident))

        offx = 1
        offy = 2.5
        beg = r"\draw[fill]"
        end = r"circle (.18);"

        for i, r in enumerate(reversed(ll)):
            for j, p in enumerate(r.pins):
                if p:
                    x = offx + .5 * j
                    y = offy + 1.5 * i
                    print("{} ({}, {}) {}".format(beg, x, y, end))

        dists = self.get_distribution(positions)
        pr = r"\node [number] at "
        for i in range(4):
            print("{}({}, -0.8) {{${}$}};".format(
                pr,
                i * 2 + 1.5, dists[0, i]))
            print("{}({}, -1.7) {{${}$}};".format(
                pr,
                i * 2 + 1.5, dists[1, i]))


class FileExistsError(click.ClickException):
    pass


class Database(object):

    def __init__(self, fname, factory=None, positions=None,
                 overwrite=False):
        if factory is not None:
            assert isinstance(factory, BoxFactory)
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
        assert isinstance(self.factory, BoxFactory)
        self.positions = attrs.positions
        self.data = h5.root.output[:]
        self.dists = self.data['dist']
        assert isinstance(self.dists, np.ndarray)
        self.ids = self.data['ident']
        assert isinstance(self.ids, np.ndarray)

        if hasattr(h5.root, 'mapping'):
            self.mapping = h5.root.mapping[:]
        h5.close()

    def save_mapping(self):
        assert hasattr(self, 'data')
        assert not hasattr(self, 'mapping')

        arr = np.zeros(len(self.data), dtype=int)
        next_group = 1
        group_dict = {}
        with click.progressbar(label="Mapping", length=len(self.data)) as bar:
            for i, dist in enumerate(self.dists):
                flat = tuple(dist.ravel())
                grp = group_dict.setdefault(flat, next_group)
                if grp == next_group:
                    next_group += 1
                arr[i] = grp
                bar.update(1)

        # Start the groupings at 0
        arr -= 1

        # Save the mapping classes
        h5 = tables.open_file(self.fname, 'r+')
        h5.create_array('/', 'mapping', arr)
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

    def make_dtype(self):
        return np.dtype([
            ('ident', int),
            ('dist', float, (len(self.positions), self.factory.output_count)),
        ])

    def from_index(self, index):
        ident = self.ids[index]
        return self.factory.from_ident(ident, index=index)

def calc_entropy(arr):
    # Extract the numpy array, and just treat it as flat. Then do the
    # standard information calculation (base 2). Deal with zeros by simply
    # cleaning up after.
    #
    q = arr.ravel()

    # Wrap this by ignoring warnings
    old_err = np.seterr(divide='ignore')
    log2 = np.where(q == 0, 0, -np.log2(q))
    # Reset warnings
    np.seterr(**old_err)

    return (q * log2).sum()


class Distribution(object):
    """Calculate some basic Info-theoretic measures on the distribution

    Note: We currently assume a uniform probability distribution across
    everything
    """

    def __init__(self, db):
        # assert isinstance(db.dists, np.ndarray)
        # assert isinstance(db.mapping, np.ndarray)
        assert len(db.dists.shape) == 3
        self.dists = db.dists.copy()
        self.mapping = db.mapping.copy()

        # First, let's make this a probability distribution assuming uniform
        # causal distributions
        sh = db.dists.shape
        normalizer = sh[0] * sh[1]
        self.dists /= normalizer
        np.testing.assert_almost_equal(self.dists.sum(), 1.0)
        self.shape = sh

        self._calc()
        self._calc_mapping()

    def _calc(self):
        """Calculating the entropies for various distributions
        """
        # Let's calculate the entropy of the three dimensions
        # 0 = P (pin layouts)
        # 1 = S (slots)
        # 2 = B (buckets)


        self.p_ent = np.log2(self.shape[0])
        self.s_ent = np.log2(self.shape[1])

        self.b_dist = self.dists.sum(axis=0).sum(axis=0)
        self.b_ent = calc_entropy(self.b_dist)

        self.p_b_dist = self.dists.sum(axis=1)
        self.p_b_ent = calc_entropy(self.p_b_dist)

        self.s_b_dist = self.dists.sum(axis=0)
        self.s_b_ent = calc_entropy(self.s_b_dist)


    def _calc_mapping(self):
        """The entropy of the mapping

        Assemble all layouts with the same mapping into bins
        Return the entropy of this (we assume a uniform dist again)
        """
        bins = np.bincount(self.mapping)
        # Normalise
        probs = bins.astype(float) / bins.sum()
        self.p_m_spec = (probs * -np.log2(probs)).sum()

    @property
    def specificity_p_for_b(self):
        return self.b_ent + self.p_ent - self.p_b_ent

    @property
    def specificity_s_for_b(self):
        return self.b_ent + self.s_ent - self.s_b_ent

    @property
    def specificity_p_for_mapping(self):
        return self.p_m_spec



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


def boxes_with_max_pins(db, found):
    best = []
    max_pins = 0
    for index in found:
        box = db.from_index(index)
        p = box.total_pins()
        if p > max_pins:
            best = [box]
            max_pins = p
        elif p == max_pins:
            best.append(box)

    return best
# ---------------------------------------------------------------------------
# Commands below here
#


@click.group()
def waddington():
    """A Python Implementation of a 'Waddington Box'
    """
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
    db = Database(FINAL_FILENAME, factory, [3, 11])
    db.save_all()
    db.save_mapping()


@waddington.command()
@click.argument('rows', default=4)
@click.argument('pins', default=15)
def quantify(rows, pins):
    """Calculate upper limit on the number of boxes given row and pin counts
    """
    rf = SpacedRowFactory(pins)
    print(len(rf.all_rows) ** rows)


@waddington.command(help="Show general info about the Box")
def describe():
    db = Database(FINAL_FILENAME)
    dst = Distribution(db)
    desc = []

    # Top row + other rows
    unconst_layouts = 2 ** (6 + (4 * 15))

    map_count = len(np.unique(db.mapping))

    def push(a, b):
        desc.append((a, b))

    push("Unconstrained Layouts", unconst_layouts)
    push("Constrained Layouts", db.factory.calc_maximum_boxes())
    push("Actual Layouts", len(db.data))
    push("Number of Mappings", map_count)
    push("Spec Slots for Buckets", dst.specificity_s_for_b)
    push("Spec Pins for Buckets", dst.specificity_p_for_b)
    push("Spec Pins for Mappings", dst.specificity_p_for_mapping)

    for text, val in desc:
        print("{0:>25}: {1:<8,}".format(text, val))


@waddington.command()
def find_lattice():
    db = Database(FINAL_FILENAME)
    q = db.dists

    # Find the high entropy distributions
    errs = np.seterr(divide='ignore')
    pmi = q * np.where(q == 0, 0, -np.log2(q))
    np.seterr(**errs)

    # Sum across layout and find the max
    summed = pmi.sum(axis=(1, 2))
    mx = summed.max()
    found = np.where(summed == mx)[0]
    print("Found {} high entropy layouts".format(len(found)))

    best = boxes_with_max_pins(db, found)
    print("Filtered to {} layouts with most pins".format(len(best)))

    # Grab the one with pins below the slots
    for b in best:
        top_row = b.layout[0].pins
        if top_row[db.positions[0]] and top_row[db.positions[1]]:
            b.dump(db.positions)


def find_pattern(target):
    db = Database(FINAL_FILENAME)
    found = np.where(np.all(db.dists == target, axis=(1, 2)))[0]
    print("Found {} candidate layouts".format(len(found)))
    best = boxes_with_max_pins(db, found)
    print("Filtered to {} layouts with most pins".format(len(best)))
    for b in best:
        b.dump(db.positions)


@waddington.command()
def find_splitfun():
    target = [[0, 1, 0, 0], [0, 0, 1, 0]]
    find_pattern(target)


@waddington.command()
def find_split():
    db = Database(FINAL_FILENAME)
    target = [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]
    found = np.where(np.all(db.dists == target, axis=(1, 2)))[0]
    print("Found {} candidate layouts".format(len(found)))
    best = []
    for ind in found:
        box = db.from_index(ind)
        frow = box.layout[0].pins
        if box.layout[-2].pins.sum() == 3 and \
                frow[db.positions[0]] and frow[db.positions[1]]:
            best.append(box)
    print("Filtered to {} candidate layouts".format(len(best)))
    for b in best:
        b.dump(db.positions)


@waddington.command()
def find_funnel():
    target = [[0, 0, 1, 0], [0, 0, 1, 0]]
    find_pattern(target)

@waddington.command()
def find_almost_switch():
    # target = [[0.0, 0.75, .25, 0], [0, 0.25, 0.75, 0]]
    # target = [[0.0, 0.75, .25, 0], [0, 0.75, 0.25, 0]]
    # target = [[0.5, 0.25, .25, 0], [0, 0.25, 0.25, 0.5]]
    target = [[1, 0, 0, 0], [0, 1, 0, 0]]
    find_pattern(target)

@waddington.command()
@click.argument('index', type=int, default=-1)
def show(index):
    """Show the layout and distribution given an index. If you don't specify
    an index, and random one will be selected.
    """
    db = Database(FINAL_FILENAME)
    if index == -1:
        index = np.random.randint(0, len(db.dists)-1)
    box = db.from_index(index)
    box.dump(db.positions)


@waddington.command()
@click.argument('index', type=int)
def tikz(index):
    """Output a tikz fragment given an index"""
    db = Database(FINAL_FILENAME)
    box = db.from_index(index)
    box.dump_tikz(db.positions)

@waddington.command()
def test():
    """Do some test operation"""
    errs = np.seterr(all='ignore')
    db = Database(FINAL_FILENAME)
    q = db.data['dist'] 
    assert isinstance(q, np.ndarray)
    over_b = q.sum(axis=1) / 2.0

    log_b = np.where(over_b == 0, 0, over_b * -np.log2(over_b))
    ent_b = log_b.sum(axis=1)

    pmi = q * np.where(q == 0, 0, -np.log2(q))
    ent_p_b = pmi.sum(axis=(1, 2))
    spec = ent_b + 1.0 - ent_p_b

    print(log_b[0], ent_p_b[0], spec[0])

    # print(ent_b.shape)
    print(ent_b.max())
    print(spec.max())

    # found = np.where(ent_b == 2.0)[0]
    # print(len(found))
    # box = db.from_index(found[0])
    # box.dump(db.positions)
    # print(ent_s.max())

    # zeros = (q == 0)

    # Find the high entropy distributions
    # pmi = q * np.where(q == 0, 0, -np.log2(q))
    np.seterr(**errs)



if __name__ == '__main__':
    waddington()
