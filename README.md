## egs-reader-for-kaldi

A python interface to read egs format data in kaldi nnet3 setup

### TODO
- [x] Implement uncompress function for CompressMatrix
- [x] Support common Matrix
- [ ] Further debug

### Support Type
* kaldi binary archieves(*.ark)
* kaldi nnet3 data examples in binary(*.egs)

### Usage

for binary *.egs:
```python
with open('10.egs', 'rb') as egs:
    while True:
        key = read_key(egs)
        if not key:
            break
        print('Egs key: {}'.format(key))
        print(read_nnet3eg(egs))
```
or 
```python
for key, eg in read_egs(ark):
    print(key)
    # ...
```
binary *.ark is similar to egs reading