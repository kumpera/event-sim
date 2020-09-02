import os;
import time;
import numpy;
from tqdm import tqdm;
import numpy as np;
import subprocess;

# for f in os.listdir('d'):
#     print(f)
if not os.access('tmp', os.R_OK):
    os.mkdir('tmp')


def zcompress_one(level):
    outfile = f'tmp/{level}.zstd'
    if os.access(outfile, os.R_OK):
        os.unlink(outfile)

    start = time.monotonic_ns()
    L = ['zstd', f'-{level}', 'extr.in', '-o', outfile, '--no-progress']
    subprocess.check_call(L, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic_ns()
    diff = (end - start) / 1_000_000
    size = os.lstat(outfile).st_size
    return (diff, size)

def zcompress_fast_one(level):
    outfile = f'tmp/{level}-fast.zstd'
    if os.access(outfile, os.R_OK):
        os.unlink(outfile)

    start = time.monotonic_ns()
    L = ['zstd', f'--fast={level}', 'extr.in', '-o', outfile, '--no-progress']
    subprocess.check_call(L, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic_ns()
    diff = (end - start) / 1_000_000
    size = os.lstat(outfile).st_size
    return (diff, size)

def gcompress_one(level):
    outfile = f'tmp/{level}.gz'
    if os.access(outfile, os.R_OK):
        os.unlink(outfile)

    start = time.monotonic_ns()
    L = ['gzip', f'-{level}', '-k', 'extr.in']
    subprocess.check_call(L, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic_ns()
    diff = (end - start) / 1_000_000
    os.rename('extr.in.gz', outfile)
    size = os.lstat(outfile).st_size
    return (diff, size)

def zdecompress_one(level):
    infile = f'tmp/{level}.zstd'
    outfile = f'tmp/ignore'
    if not os.access(infile, os.R_OK):
        raise Exception(f'missing input file {infile}')
    if os.access(outfile, os.R_OK):
        os.unlink(outfile)

    start = time.monotonic_ns()
    L = ['zstd', '-d', infile, '-o', outfile, '--no-progress']
    subprocess.check_call(L, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic_ns()
    diff = (end - start) / 1_000_000
    return diff

def zdecompress_fast_one(level):
    infile = f'tmp/{level}-fast.zstd'
    outfile = f'tmp/ignore'
    if not os.access(infile, os.R_OK):
        raise Exception(f'missing input file {infile}')
    if os.access(outfile, os.R_OK):
        os.unlink(outfile)

    start = time.monotonic_ns()
    L = ['zstd', '-d', infile, '-o', outfile, '--no-progress']
    subprocess.check_call(L, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic_ns()
    diff = (end - start) / 1_000_000
    return diff
def gdecompress_one(level):
    infile = f'tmp/{level}.gz'
    outfile = f'tmp/{level}'
    if not os.access(infile, os.R_OK):
        raise Exception(f'missing input file {infile}')
    if os.access(outfile, os.R_OK):
        os.unlink(outfile)

    start = time.monotonic_ns()
    L = ['gunzip', '-k', infile]
    subprocess.check_call(L, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic_ns()
    diff = (end - start) / 1_000_000
    return diff

def run_decompress_bench(decompress, levels, time_budget_in_secs):
    base_size = os.lstat('extr.in').st_size
    results = []
    for level in tqdm(levels):
        level_end = time.time() + time_budget_in_secs
        dtime = []
        while time.time() < level_end:
            res = decompress(level)
            dtime.append(res)

        results.append([level, len(dtime), np.mean(dtime), np.std(dtime), np.percentile(dtime, 25, interpolation = 'midpoint'), np.percentile(dtime, 75, interpolation = 'midpoint') ])
    return results


def run_compress_bench(compress, levels, time_budget_in_secs):
    base_size = os.lstat('extr.in').st_size
    results = []
    for level in tqdm(levels):
        level_end = time.time() + time_budget_in_secs
        ctime = []
        ratio = 0
        while time.time() < level_end:
            res = compress(level)
            ctime.append(res[0])
            ratio = base_size / res[1]

        results.append([level, ratio, len(ctime), np.mean(ctime), np.std(ctime), np.percentile(ctime, 25, interpolation = 'midpoint'), np.percentile(ctime, 75, interpolation = 'midpoint') ])
    return results

def dump_array(outfile, header, arr):
    for l in arr:
        outfile.write(f'{header},')
        outfile.write(",".join([str(i) for i in l]))
        outfile.write('\n')


TIME_PER_LEVEL = 60
algos = [
    ['zstd', list(range(0, 11)), zcompress_one, zdecompress_one],
    ['gzip', list(range(1, 10)), gcompress_one, gdecompress_one],
    ['zfast', list(range(1, 10)), zcompress_fast_one, zdecompress_fast_one],
]

TIME_PER_LEVEL = 100
suffix = '-3'

# with open(f'compress{suffix}.csv', 'w') as stats:
#     stats.write('method,level,ratio,samples,mean,std,p25,p75\n')
#     for a in algos:
#         dump_array(stats, a[0], run_compress_bench(a[2], a[1], TIME_PER_LEVEL))

with open(f'decompress{suffix}.csv', 'w') as stats:
    stats.write('method,level,samples,mean,std,p25,p75\n')
    for a in algos:
        dump_array(stats, a[0], run_decompress_bench(a[3], a[1], TIME_PER_LEVEL))

