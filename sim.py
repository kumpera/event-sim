import argparse;
import numpy as np;
import zlib;
import zstandard as zstd;
import brotli;
import snappy;
import json;
import random;
from tqdm import tqdm;

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
        raise Exception("missing method process")

    #called before anything item is processed
    def on_batch_start(self):
        pass

    # called after the last item was processed before the batch ends
    def on_batch_end(self):
        pass

    def reprocess_across_batches(self):
        return False
    
    def gen_specific_csv(self):
        return None

    def get_header(self):
        return None

    def add_header_bytes(self, header_size):
        self.cur_batch_header_size += header_size
        self.cur_batch_size += header_size
    
    def finish_batch(self):
        self.on_batch_end()

        self.batches.append([self.cur_batch_size, self.cur_batch_raw_size, self.cur_batch_line_count, self.cur_batch_header_size])

        self.cur_batch_size = 0
        self.cur_batch_raw_size = 0
        self.cur_batch_line_count = 0
        self.cur_batch_header_size = 0

        self.on_batch_start()

    def does_item_overflow(self, item_size, cur_size, max_size):
        return item_size + cur_size > max_size

    def start(self):
        self.on_batch_start()

    def add_bytes(self, line):
        original_len = len(line)
        item_size = self.process(line)
        if self.does_item_overflow(item_size, self.cur_batch_size, self.max_batch_size):
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

    def gen_csv(self, header, out_file):
        n = np.array(self.batches)
        #label, n_batches, mean_ratio, mean_lines, mean_header, mean_size
        generic_line = f'{self.label},{len(self.batches)},{np.mean(n[:,1] / n[:,0])},{np.mean(n[:,2])},{np.mean(n[:,3])},{np.mean(n[:,0])}'
        specific = self.gen_specific_csv()
        line = f'{header},{generic_line}'
        if specific != None:
            line = f'{line},{specific}'
        out_file.write(line)
        out_file.write('\n')


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

class Dedup(AccumulateBatch):
    def __init__(self, label, max_dict_size, max_batch_size):
        super().__init__(f'dedup-{label}_{max_dict_size}', max_batch_size)
        self.max_dict_size = max_dict_size
        self.action_set = dict()
        self.cur_dict = dict()
        self.hits = 0
        self.misses = 0
        self.dedup_batch_stats = []

    def get_header(self):
        return 'mean-dict-lines,mean-dict-hit-ratio,mean-action-hit-ratio'

    def register_new_action(self, action, json_action):
       raise Exception("must override")

    def build_dict_from_prev_batch(self):
        return True

    def gen_specific_csv(self):
        # mean-dict-lines, mean-dict-hit-ratio, mean-action-hit-ratio
        n = np.array(self.dedup_batch_stats)
        mean_dict_lines = np.mean(n[:,0])
        mean_dict_hit_ratio = np.mean(n[:,1] / n[:,0])
        mean_action_hit_ratio = np.mean(n[:,2] / (n[:,2] + n[:,3]))

        return f'{mean_dict_lines},{mean_dict_hit_ratio},{mean_action_hit_ratio}'

    def batch_done(self, batch_lines):
        if self.build_dict_from_prev_batch() == False:
            # used entries will always be equal to cur_dict
            log_line = [len(self.cur_dict), len(self.cur_dict), self.hits, self.misses]
            self.dedup_batch_stats.append(log_line)
            self.hits = self.misses = 0
            self.cur_dict = dict()
            return

        # this is called after the batch is done
        lst = list(self.action_set.items())
        lst.sort(key=lambda x: x[1] * len(x[0]), reverse=True)
        final_dict = dict()
        total_len = 0
        actions = 0

        # don't record stats for first batch as it won't have a dict
        if len(self.cur_dict) > 0:
            found_actions = set(self.action_set.keys())
            used_entries = 0
            for k in self.cur_dict:
                if k in found_actions:
                    used_entries += 1

            log_line = [len(self.cur_dict), used_entries, self.hits, self.misses]
            self.dedup_batch_stats.append(log_line)

        for kv in lst:
            if total_len >= self.max_dict_size:
                break
            final_dict[kv[0]] = f'id_{random.randint(0, 1_000_000_000)}'
            total_len += len(kv[0])
            actions += 1

        self.cur_dict = final_dict
        self.action_set = dict()
        self.hits = self.misses = 0

    def on_batch_start(self):
        if len(self.cur_dict) > 0 and self.build_dict_from_prev_batch():
            self.add_header_bytes(self.process_header(json.dumps(self.cur_dict)))

    def on_batch_end(self):
       if self.build_dict_from_prev_batch() == False:
           self.add_header_bytes(self.process_header(json.dumps(self.cur_dict)))


    def process(self, data):
        # print(f'cur entries: {len(self.cur_dict)}')

        evt = json.loads(data.decode('utf-8'))
        actions2 = []
        for action in evt["c"]["_multi"]:
            x = json.dumps(action)
            if x in self.cur_dict:
                self.hits += 1
                actions2.append({ '__idx': self.cur_dict[x]})
            else:
                self.misses += 1
                if self.build_dict_from_prev_batch():
                    actions2.append(action)
                else:
                    actions2.append(self.register_new_action(x, action))

            if self.build_dict_from_prev_batch():
                if x in self.action_set:
                    self.action_set[x] += 1
                else:
                    self.action_set[x] = 1

            evt["c"]["_multi"] = actions2

        new_dump = json.dumps(evt)
        return self.process_transformed_event(new_dump)

class DedupSimple(Dedup):
    def __init__(self, max_dict_size, max_batch_size):
        super().__init__('simple', max_dict_size, max_batch_size)

    def process_header(self, dict_dump):
        return len(dict_dump)

    def process_transformed_event(self, line):
        return len(line)

class DedupZstd(Dedup):
    def __init__(self, params, max_batch_size):
        self.level = params[0]
        self.max_dict_size = params[1]
        super().__init__(f'zstd_{self.level}', self.max_dict_size, max_batch_size)

    def process_header(self, dict_dump):
        data = bytes(dict_dump, 'utf-8')
        return len(zstd.ZstdCompressor(level=self.level).compress(data))
        return res

    def process_transformed_event(self, line):
        data = bytes(line, 'utf-8')
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

class DedupZstd2(Dedup):
    def __init__(self, params, max_batch_size):
        self.level = params[0]
        self.max_dict_size = params[1]
        super().__init__(f'zstd2_{self.level}', self.max_dict_size, max_batch_size)
        self.current_dict_size = 0
        self.pending_actions = []

    def build_dict_from_prev_batch(self):
        return False

    def does_item_overflow(self, item_size, cur_size, max_size):
        # we use this to estimate how big the dictionary will compress to. (we use a static 2.8 compression ratio)
        estimated_dict_size = self.current_dict_size / 2.8
        res = item_size + estimated_dict_size + cur_size > max_size
        if not res:
            for p in self.pending_actions:
                self.cur_dict[p[0]] = p[1]
            self.pending_actions = []
        return res

    def register_new_action(self, action, json_action):
        self.current_dict_size += len(action)
        action_id = f'id_{random.randint(0, 1_000_000_000)}'
        # we want to avoid commiting new entries to the dictionary if we don't have to
        self.pending_actions.append([action, action_id])
        return { '__idx': action_id }

    def process_transformed_event(self, line):
        data = bytes(line, 'utf-8')
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

    def process_header(self, dict_dump):
        data = bytes(dict_dump, 'utf-8')
        return len(zstd.ZstdCompressor(level=self.level).compress(data))
        return res

    def on_batch_end(self):
        super().on_batch_end()
        self.current_dict_size = 0
        self.pending_actions = []


class DedupZstd3(Dedup):
    def __init__(self, params, max_batch_size):
        self.level = params[0]
        self.max_dict_size = params[1]
        super().__init__(f'zstd3_{self.level}', self.max_dict_size, max_batch_size)

    def build_dict_from_prev_batch(self):
        return False

    def register_new_action(self, action, json_action):
        action_id = f'id_{random.randint(0, 1_000_000_000)}'
        # we want to avoid commiting new entries to the dictionary if we don't have to
        self.cur_dict[action] = action_id
        return json_action

    def process_transformed_event(self, line):
        data = bytes(line, 'utf-8')
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

    def process_header(self, dict_dump):
        # data = bytes(dict_dump, 'utf-8')
        # return len(zstd.ZstdCompressor(level=self.level).compress(data))
        # return res
        return 0


class DedupZstdDict(Dedup):
    def __init__(self, params, max_batch_size):
        self.level = params[0]
        self.max_dict_size = params[1]
        self.max_zdict_size = params[2]
        self.zdict_lines = []
        self.cur_zdict = None
        super().__init__(f'zstd-dict_{self.level}_{self.max_zdict_size}', self.max_dict_size, max_batch_size)

    def batch_done(self, batch_lines):
        super().batch_done(batch_lines)
        self.cur_zdict = zstd.train_dictionary(self.max_zdict_size, self.zdict_lines)
        self.zdict_lines = []

    def compress_and_log(self, text_data):
        data = bytes(text_data, 'utf-8')
        self.zdict_lines.append(data)
        if self.cur_zdict != None:
            return len(zstd.ZstdCompressor(level=self.level, dict_data=self.cur_zdict).compress(data))
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

    def process_header(self, dict_dump):
        dedup_dict_size = self.compress_and_log(dict_dump)
        dict_bytes = self.cur_zdict.as_bytes()
        comp_dict_size = len(zstd.ZstdCompressor(level=self.level).compress(dict_bytes))
        return dedup_dict_size + comp_dict_size

    def process_transformed_event(self, line):
        return self.compress_and_log(line)

#don't zdict compress the actions dict
class DedupZstdDict2(Dedup):
    def __init__(self, params, max_batch_size):
        self.level = params[0]
        self.max_dict_size = params[1]
        self.max_zdict_size = params[2]
        self.zdict_lines = []
        self.cur_zdict = None
        super().__init__(f'zstd-dict2_{self.level}_{self.max_zdict_size}', self.max_dict_size, max_batch_size)

    def batch_done(self, batch_lines):
        super().batch_done(batch_lines)
        self.cur_zdict = zstd.train_dictionary(self.max_zdict_size, self.zdict_lines)
        self.zdict_lines = []

    def compress_and_log(self, text_data, use_dict):
        data = bytes(text_data, 'utf-8')
        if use_dict:
            self.zdict_lines.append(data)
        if self.cur_zdict != None and use_dict:
            return len(zstd.ZstdCompressor(level=self.level, dict_data=self.cur_zdict).compress(data))
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

    def process_header(self, dict_dump):
        dedup_dict_size = self.compress_and_log(dict_dump, False)
        dict_bytes = self.cur_zdict.as_bytes()
        comp_dict_size = len(zstd.ZstdCompressor(level=self.level).compress(dict_bytes))
        return dedup_dict_size + comp_dict_size

    def process_transformed_event(self, line):
        return self.compress_and_log(line, True)

#zdict compress only the actions dict
class DedupZstdDict3(Dedup):
    def __init__(self, params, max_batch_size):
        self.level = params[0]
        self.max_dict_size = params[1]
        self.max_zdict_size = params[2]
        self.cur_zdict = None
        super().__init__(f'zstd-dict3_{self.level}_{self.max_zdict_size}', self.max_dict_size, max_batch_size)

    def process_header(self, dict_dump):
        train_data = []
        for k in self.cur_dict:
            train_data.append(bytes(k, 'utf-8'))
        # train on each action independently
        self.cur_zdict = zstd.train_dictionary(self.max_zdict_size, train_data)

        data = bytes(dict_dump, 'utf-8')
        dedup_dict_size = len(zstd.ZstdCompressor(level=self.level, dict_data = self.cur_zdict).compress(data))

        dict_bytes = self.cur_zdict.as_bytes()
        comp_dict_size = len(zstd.ZstdCompressor(level=self.level).compress(dict_bytes))
        return dedup_dict_size + comp_dict_size

    def process_transformed_event(self, line):
        data = bytes(line, 'utf-8')
        return len(zstd.ZstdCompressor(level=self.level).compress(data))

# zstd-dict only mode
class ZstdDict(AccumulateBatch):
    def __init__(self, params, max_batch_size):
        super().__init__(f'zstd-dict_{params[0]}_{params[1]}', max_batch_size)
        self.level = params[0]
        self.train_dict_size = params[1]
        self.cur_dict = None
        self.acc_lines = []

    def batch_done(self, batch_lines):
        self.acc_lines.extend(batch_lines)
        # we truncate before training as in practice we'd keep a size limited history 
        if len(self.acc_lines) > 400:
            self.acc_lines = self.acc_lines[-400:]

        self.cur_dict = zstd.train_dictionary(self.train_dict_size, self.acc_lines)

    def on_batch_start(self):
        if self.cur_dict != None:
            dict_bytes = self.cur_dict.as_bytes()
            comp_dict = zstd.ZstdCompressor(level=self.level).compress(dict_bytes)
            self.add_header_bytes(len(comp_dict))

    def process(self, data):
        if self.cur_dict == None:
            return len(zstd.ZstdCompressor(level=self.level).compress(data))

        res = len(zstd.ZstdCompressor(level=self.level, dict_data=self.cur_dict).compress(data))
        # print(f'{self.label} :: {len(self.batches)} :: {res}' )
        return res

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

    def gen_csv(self, file_name, out_file):
        header = f'{file_name},{self.id},{self.lines},{self.raw_size}'
        for p in self.procs:
            p.gen_csv(header, out_file)

    def finish(self):
        print(f'client_{self.id} lines:{self.lines} raw-size:{self.raw_size}')
        for p in self.procs:
            p.report()


parser = argparse.ArgumentParser(description="Compression simulation")
parser.add_argument('files', nargs='+', help='Log files to use')
parser.add_argument('--clients', '-c', type=int, help='Number of clients to use (default 1)', default=1)
parser.add_argument('--algo', nargs='?', 
    help='Which compression algorithms to try: zlib, zstd, brotli, snappy, zstd-dict,dedup (default zstd)', default='zstd')
parser.add_argument('--sweep', help="Param sweep the current best know algo (dedup_zstd)", default=False, action='store_true')
parser.add_argument('--sweep2', help="Sweep top two (dedup_zstd and zstd-dict) with a reasonable grid", default=False, action='store_true')
parser.add_argument('--csv', help="Gen stats in csv form", default=False, action='store_true')
parser.add_argument('--prefix', help="Prefix for output files", default='')

args = parser.parse_args()
algo_names = args.algo.split(',')
#TODO make it configurable
MAX_BATCH_SIZE = 198 * 1024

compression_algos = {
    'zlib': [Deflate, 1, -1, 9],
    'zstd': [Zstd, -1, 0, 19],
    'zstd-dict':[ZstdDict, [13, 220_000], [13, 140_000] ],
    'brotli': [Brotli, 0, 3, 11],
    'snappy': [Snappy],
    'dedup': [DedupSimple, 10_000, 20_000, 60_000, 100_000],
    'dedup-zstd': [DedupZstd, [1, 200_000], [13, 200_000] ],
    # 'dedup-zstd2': [DedupZstd2, [1, 200_000], [13, 200_000] ],
    'dedup-zstd2': [DedupZstd2, [10, 200_000] ],
    # 'dedup-zstd3': [DedupZstd3, [1, 200_000], [13, 200_000] ],
    'dedup-zstd3': [DedupZstd3, [10, 200_000] ],
    'dedup-zstd-dict': [DedupZstdDict, [10, 100_000, 100_000], [10, 100_000, 200_000], [10, 200_000, 100_000], [10, 200_000, 200_000]],
    'dedup-zstd-dict2': [DedupZstdDict2, [10, 100_000, 100_000], [10, 100_000, 200_000], [10, 200_000, 100_000], [10, 200_000, 200_000]],
    'dedup-zstd-dict3': [DedupZstdDict3, [10, 200_000, 10_000], [10, 200_000, 20_000], [10, 200_000, 40_000], [10, 240_000, 40_000], [19, 200_000, 20_000]],
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

def gen_sweep_list():
    res = []
    # 240_000 is the previously know best dict size
    for level in range(0, 19):
        res.append(DedupZstd2([level, 240_000], MAX_BATCH_SIZE))
    
    # 13 is the previously best well known compression level
    for i in range(8, 30):
        res.append(DedupZstd2([13, i * 10_000], MAX_BATCH_SIZE))
    return res

def gen_sweep_list2():
    res = []
    for l in [1, 13]:
        for max_dict in [80_000, 160_000, 240_000]:
            res.append(DedupZstd([l, max_dict], MAX_BATCH_SIZE))
            res.append(ZstdDict([l, max_dict], MAX_BATCH_SIZE))
            res.append(DedupZstdDict3([l, max_dict, max_dict], MAX_BATCH_SIZE))
    return res


def gen_clients(args):
    clients = []
    for i in range(0, args.clients):
        c = Client(i)
        if args.sweep:
            for p in gen_sweep_list():
                c.add_proc(p)
        elif args.sweep2:
            for p in gen_sweep_list2():
                c.add_proc(p)
        else:
            for p in gen_compression_list(algo_names):
                c.add_proc(p)
        clients.append(c)
    return clients

for cur_file in tqdm(args.files):
    with open(cur_file, 'r+') as input:
        c = gen_clients(args)[0]
        c.start()
        cur_line = 0
        for line in tqdm(input.readlines()):
            c.add_line(line)

    if args.csv:
        with open(f'{args.prefix}{cur_file}.csv', 'w') as stats:
            stats.write('file,client,total_lines,raw_size,name,n_batches,mean_ratio,mean_lines,mean_header,mean_batch_size')
            if c.procs[0].get_header() != None:
                stats.write(f',{c.procs[0].get_header()}')
            stats.write('\n')
            c.gen_csv(cur_file, stats)
    else:
        c.finish()

