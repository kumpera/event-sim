import argparse;
import numpy as np;
import zlib;
import zstandard as zstd;

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

args = parser.parse_args()

#TODO make it configurable
MAX_BATCH_SIZE = 198 * 1024

clients = []
for i in range(0, args.clients):
    c = Client(i)
    c.add_proc(Deflate(1, MAX_BATCH_SIZE))
    c.add_proc(Deflate(-1, MAX_BATCH_SIZE)) #default
    c.add_proc(Deflate(9, MAX_BATCH_SIZE))

    c.add_proc(Zstd(-1, MAX_BATCH_SIZE))
    c.add_proc(Zstd(0, MAX_BATCH_SIZE)) #default
    c.add_proc(Zstd(19, MAX_BATCH_SIZE))

    clients.append(c)


with open(args.files[0], 'r+') as input:
    for line in input.readlines():
        clients[0].add_line(line)

clients[0].finish()
