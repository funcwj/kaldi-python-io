## Kaldi Python IO

A python (3.6+) wrapper for kaldi's data accessing.

### Support Type

* kaldi's binary archives (*.ark)
* kaldi's scripts (alignments & features, *.scp)
* kaldi nnet3 data examples in binary (*.egs)

### Install

`python setup.py install` or `pip install kaldi_python_io`

### Usage

* ArchiveReader && AlignArchiveReader
    ```python
    # allow only sequential index
    ark_reader = ArchiveReader("copy-feats ark:foo.ark ark:- |", matrix=True)
    for key, _ in ark_reader:
        print(key)
    ali_reader = AlignArchiveReader("gunzip -c foo.ali.gz |")
    for key, _ in ark_reader:
        print(key)
    ```

* Nnet3EgsReader
    ```python
    # allow only sequential index
    egs_reader = Nnet3EgsReader("foo.egs")
    for key, _ in egs_reader:
        print("{}".format(key))
    ```

* ArchiveWriter
    ```python
    with ArchiveWriter("foo.ark", "foo.scp") as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    ```

* ScriptReader && AlignScriptReader
    ```python
    # allow sequential/random index
    scp_reader = ScriptReader("shuf foo.scp | head -n 2", matrix=True)
    for key, mat in scp_reader:
        print("{}: {}".format(key, mat.shape))
    ali_reader = AlignScriptReader("foo.ali.scp")
    for key, ali in ali_reader:
        print("{}: {}".format(key, ali.shape))
    ```
