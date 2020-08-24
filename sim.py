import argparse;
import numpy as np;
import zlib;
import zstandard as zstd;
import brotli;
import snappy;

class LineProcessor:
    def __init__(self, label, max_batch_size):
        self.label = label
        self.max_batch_size = max_batch_size

        self.cur_batch_size = 0
        self.cur_batch_raw_size = 0
        self.cur_batch_line_count = 0
        self.cur_batch_header_size = 0
        self.batches = []

    def process(self, data):
        raise "missing"

    def on_batch_start(self):
        pass

    def reprocess_across_batches(self):
        return False

    def add_header_bytes(self, header_size):
        self.cur_batch_header_size += header_size
        self.cur_batch_size += header_size
    
    def finish_batch(self):
        self.batches.append([self.cur_batch_size, self.cur_batch_raw_size, self.cur_batch_line_count, self.cur_batch_header_size])

        self.cur_batch_size = 0
        self.cur_batch_raw_size = 0
        self.cur_batch_line_count = 0
        self.cur_batch_header_size = 0

        self.on_batch_start()

    def start(self):
        self.on_batch_start()

    def add_bytes(self, line):
        original_len = len(line)
        item_size = self.process(line)
        if item_size + self.cur_batch_size > self.max_batch_size:
            self.finish_batch()
        if self.reprocess_across_batches():
            item_size = self.process(line)

        self.cur_batch_size += item_size
        self.cur_batch_raw_size += original_len
        self.cur_batch_line_count += 1

    def report(self):
        print(f'{self.label} batches:{len(self.batches)}')
        n = np.array(self.batches)
        mean_ratio = np.mean(n[:,1] / n[:,0])
        print(f'\tmean-ratio: {mean_ratio:2.2f} mean-lines:{np.mean(n[:,2]):.1f} mean-header:{int(np.mean(n[:,3]))}')

class AccumulateBatch(LineProcessor):
    def __init__(self, label, max_batch_size):
        super().__init__(label, max_batch_size)
        self.batch_data = []

    def add_bytes(self, line):
        super().add_bytes(line)
        self.batch_data.append(line)

    def batch_done(self, batch_lines):
        pass

    def reprocess_across_batches(self):
        return True

    def finish_batch(self):
        super().finish_batch()
        self.batch_done(self.batch_data)
        self.batch_data = []


class ZstdDict(AccumulateBatch):
    def __init__(self, params, max_batch_size):
        super().__init__(f'zstd_dict_{params[0]}_{params[1]}', max_batch_size)
        self.level = params[0]
        self.train_dict_size = params[1]
        self.cur_dict = None

    def batch_done(self, batch_lines):
        self.cur_dict = zstd.train_dictionary(self.train_dict_size, batch_lines)

    def on_batch_start(self):
        if self.cur_dict != None:
            dict_bytes = self.cur_dict.as_bytes()
            comp_dict = zstd.ZstdCompressor(level=self.level).compress(dict_bytes)
            self.add_header_bytes(len(comp_dict))

    def process(self, data):
        if self.cur_dict == None:
            return len(zstd.ZstdCompressor(level=self.level).compress(data))
        return len(zstd.ZstdCompressor(level=self.level, dict_data=self.cur_dict).compress(data))


# Line-a-time compression algos (maybe making them not a subclass of LineProcessor would make it easy to share?)
class Deflate(LineProcessor):
    def __init__(self, level, max_batch_size):
        super().__init__(f'zlib_{level}', max_batch_size)
        self.level = level
    
    def process(self, data):
        return len(zlib.compress(data, level=self.level))

class Zstd(LineProcessor):
    def __init__(self, level, max_batch_size):
        super().__init__(f'zstd_{level}', max_batch_size)
        self.level = level
    
    def process(self, data):
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

class Brotli(LineProcessor):
    def __init__(self, level, max_batch_size):
        super().__init__(f'brotli_{level}', max_batch_size)
        self.level = level
    
    def process(self, data):
        return len(brotli.compress(data, quality=self.level))

class Snappy(LineProcessor):
    def __init__(self, max_batch_size):
        super().__init__('snappy', max_batch_size)
    
    def process(self, data):
        return len(snappy.compress(data))

class Client:
    def __init__(self, id):
        self.id = id
        self.procs = []
        self.lines = 0
        self.raw_size = 0

    def add_proc(self, proc):
        self.procs.append(proc)

    def add_line(self, line):
        self.lines += 1
        self.raw_size += len(line)
        data = bytes(line, 'utf-8')
        for p in self.procs:
            p.add_bytes(data)

    def start(self):
        for p in self.procs:
            p.start()

    def finish(self):
        print(f'client_{self.id} lines:{self.lines} raw-size:{self.raw_size}')
        for p in self.procs:
            p.report()


parser = argparse.ArgumentParser(description="Compression simulation")
parser.add_argument('files', nargs='+', help='Log files to use')
parser.add_argument('--clients', '-c', type=int, help='Number of clients to use (default 1)', default=1)
parser.add_argument('--algo', nargs='?', 
    help='Which compression algorithms to try: zlib, zstd, brotli, snappy, zstd-dict (default zstd)', default='zstd')
# compression dimention

args = parser.parse_args()
algo_names = args.algo.split(',')
#TODO make it configurable
MAX_BATCH_SIZE = 198 * 1024

compression_algos = {
    'zlib': [Deflate, 1, -1, 9],
    'zstd': [Zstd, -1, 0, 19],
    'zstd-dict':[ZstdDict, [0, 10_000], [0, 20_000], [0, 40_000], [0, 80_000], [0, 120_000], [10, 100_000], [19, 160_000] ],
    'brotli': [Brotli, 0, 3, 11],
    'snappy': [Snappy],
}

def gen_compression_list(algos):
    res = []
    for name in algos:
        x = compression_algos[name]
        if len(x) == 1:
            res.append(x[0](MAX_BATCH_SIZE))
        else:
            for i in range(1, len(x)):
                res.append(x[0](x[i], MAX_BATCH_SIZE))
    return res

clients = []
for i in range(0, args.clients):
    c = Client(i)
    for p in gen_compression_list(algo_names):
        c.add_proc(p)

    clients.append(c)


with open(args.files[0], 'r+') as input:
    clients[0].start()
    for line in input.readlines():
        clients[0].add_line(line)

clients[0].finish()
