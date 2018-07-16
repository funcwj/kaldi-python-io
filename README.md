## kaldi IO

A python wrapper for kaldi's data accessing.

### Support Type

* kaldi's binary archieves(*.ark)
* kaldi's scripts (alignments & features, *.scp)
* kaldi nnet3 data examples in binary(*.egs)

### Usage

* ArchieveReader & AlignmentReader
    ```python
    # allow sequential/random index
    ark_reader = ArchieveReader("egs.scp")
    for key, _ in ark_reader:
        print(key)
    ali_reader = AlignmentReader("ali.scp")
    for key, _ in ali_reader:
        print(key)
    ```

* ArchieveWriter
    ```python
    with ArchieveWriter("egs.ark", "egs.scp") as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    ```