# Overview

This repo is a collection of scripts to simulate and benchmark the many approaches to compress logs.

# Instalation

```
pip install numpi brotlipy python-snappy zstandard
```

# TODO

All those items are suggestions based on what we find to be useful.


## Intra-batch techniques

- Add more compression methods for per-event compression
    * DONE zlib, zstd, brotli and snappy.
- Compression dict trained on previous batch(es).
    * DONE for zstd
- Action dedup with dictionary trained on previous batch(es)
    * DONE
- Add compression on top of action dedup
    * DONE
- double dictionary mode trained on previous batches
    * DONE
- double dictionary mode adaptatively trained on current payload
    * SKIP double dictionary is not promissing

## dedup-zdict experiments

It's clear from the previous batch thay `dedup-zdict` is the best algo to go with.
It provides the best compression even in the 1k actions scenario that was supposed
to favor the trained dictionary approach.

### Findings and future work needed:

- Compression doesn't matter at high hit ratios
- Dictionary size hurts us at low hit ratios
- Compression quality matters at low hit ratios
- Dictionary size should be adaptative
- Building the dictionary from previous batches is not optimal
- Naive criteria for including an action into the dictionary


### Reproducing

1) Run the following:
```
python gen-batches.py --name batch --set1
python sim.py --sweep2 --csv batch_*
```

2) Open many-algos-ratios.ipynb and evaluate it


# Statistics
- Compression and decompression times
- Multiple knobs - payload size, dict sizes, etc
- number of batch overflow bytes (IE, how many bytes we went over the batch limit)
- min/max/stddev
- csv output good for ploting/spreadshet

# Inter-batch techniques

- rolling action dedup
- static dictinaries with explicit references
- one client vs multiple clients

# Inter-batch failure modes

- client restart
- partition failure

# Misc

- Make batch size configurable
- better handle multi-dimentional arguments (see ZstdDict)
- add param sweep with hill climb