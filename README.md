## kaldi IO

A python wrapper for kaldi's data accessing.

### Support Type

* kaldi's binary archives(*.ark)
* kaldi's scripts (alignments & features, *.scp)
* kaldi nnet3 data examples in binary(*.egs)

### Usage

* ArchiveReader
    ```python
    # allow only sequential index
    ark_reader = ArchiveReader("egs.ark")
    for key, _ in ark_reader:
        print(key)
    ```

* Nnet3EgsReader
    ```python
    # allow only sequential index
    egs_reader = Nnet3EgsReader(egs)
    for key, _ in egs_reader:
        print("{}".format(key))
    ```

* ArchiveWriter
    ```python
    with ArchiveWriter("egs.ark", "egs.scp") as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    ```

* ScriptReader && AlignmentReader
    ```python
    # allow sequential/random index
    scp_reader = ScriptReader("egs.scp")
    for key, mat in scp_reader:
        print("{}: {}".format(key, mat.shape))
    ali_reader = AlignmentReader("ali.scp")
    for key, ali in ali_reader:
        print("{}: {}".format(key, ali.shape))
    ```
