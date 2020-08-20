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
- Add compression on top of action dedup
- double dictionary mode trained on previous batches
- double dictionary mode adaptatively trained on current payload

# Statistics
et
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