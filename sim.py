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
        self.batches = []

    def process(self, data):
        return len(data)
    
    def finish_batch(self):
        self.batches.append([self.cur_batch_size, self.cur_batch_raw_size, self.cur_batch_line_count])

        self.cur_batch_size = 0
        self.cur_batch_raw_size = 0
        self.cur_batch_line_count = 0

    def add_bytes(self, line):
        original_len = len(line)
        item_size = self.process(line)
        if item_size + self.cur_batch_size > self.max_batch_size:
            self.finish_batch()

        self.cur_batch_size += item_size
        self.cur_batch_raw_size += original_len
        self.cur_batch_line_count += 1

    def report(self):
        print(f'{self.label} batches:{len(self.batches)}')
        n = np.array(self.batches)
        mean_ratio = np.mean(n[:,1] / n[:,0])
        print(f'\tmean-ratio: {mean_ratio:2.2f} mean-lines:{np.mean(n[:,2])}')


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

    def finish(self):
        print(f'client_{self.id} lines:{self.lines} raw-size:{self.raw_size}')
        for p in self.procs:
            p.report()


parser = argparse.ArgumentParser(description="Compression simulation")
parser.add_argument('files', nargs='+', help='Log files to use')
parser.add_argument('--clients', '-c', type=int, help='Number of clients to use (default 1)', default=1)
parser.add_argument('--algo', nargs='?', 
    help='Which compression algorithms to try: zlib, zstd, brotli, snappy (default zstd)', default='ztd')
# compression dimention

args = parser.parse_args()
algo_names = args.algo.split(',')
#TODO make it configurable
MAX_BATCH_SIZE = 198 * 1024

compression_algos = {
    'zlib': [Deflate, 1, -1, 9],
    'zstd': [Zstd, -1, 0, 19],
    'brotli': [Brotli, 0, 3, 11],
    'snappy': [Snappy]
}

def gen_compression_list(algos):
    res = []
    for name in algos:
        x = compression_algos[name]
        if len(x) == 1:
            res.append(x[0](MAX_BATCH_SIZE))
        else:
            res.append(x[0](x[1], MAX_BATCH_SIZE))
            res.append(x[0](x[2], MAX_BATCH_SIZE))
            res.append(x[0](x[3], MAX_BATCH_SIZE))
    return res

clients = []
for i in range(0, args.clients):
    c = Client(i)
    for p in gen_compression_list(algo_names):
        c.add_proc(p)
    # c.add_proc(Deflate(1, MAX_BATCH_SIZE))
    # c.add_proc(Deflate(-1, MAX_BATCH_SIZE)) #default
    # c.add_proc(Deflate(9, MAX_BATCH_SIZE))

    # c.add_proc(Zstd(-1, MAX_BATCH_SIZE))
    # c.add_proc(Zstd(0, MAX_BATCH_SIZE)) #default
    # c.add_proc(Zstd(19, MAX_BATCH_SIZE))

    clients.append(c)


with open(args.files[0], 'r+') as input:
    for line in input.readlines():
        clients[0].add_line(line)

clients[0].finish()
