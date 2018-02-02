## kaldi IO

A python wrapper for kaldi's data accessing.

### Support Type

* kaldi's binary archieves(*.ark)
* kaldi's scripts (alignments & features, *.scp)
* kaldi nnet3 data examples in binary(*.egs)

### Usage

* ScpReader
```
scp_reader = ScpReader('data/dev_feats.scp')
for key, ark in scp_reader:
    print(ark.shape)
    assert key in scp_reader
```

* ArkReader
```
ark_reader = ArkReader('../data/pdf.*.ark', model='pdfid')
for key, vec in ark_reader:
    print(key)
```

more to see `_test_*` function in `iobase.py`, `reader.py`