# FSM-Benchmarking

## Generating Random FSTs

Open utils/ramdomfstdb.jl and change the variables on top to design the type of randomfsts to be generated
(TODO: create command line arguments). The run the command and it will generate a folder with machines in binary format and a csv with some information.

```
julia utils/ramdomfstdb.jl
```

## Composition

For composition we will try composing the machines generated in the folder with this command.

```
julia composition/composition_fsadb.jl path_to_folder
```

This will generate a new folder that ends in "_composed" and a csv file which we will have the compositions that succeded.

For running the benchmarks, open the script runcompbench.sh and change the path to the julia project folder and to the folder with the machine (with out the _composed)

```
runcompbench.sh
```

### Benchmark real example of utterance composed with phone LM

```
fstcompose ashrafsamplesent.noeps.fst voxpopuli_ipatranscription.5.fst comp.fst
```

Links:

[utterance dense fst](https://www.dropbox.com/s/ni65bktfcs7y4iw/ashrafsamplesent.noeps.fst?dl=0)

[phone lm](https://www.dropbox.com/s/dzvq869dr63r5j5/voxpopuli_ipatranscription.5.fst?dl=0)

[results](https://www.dropbox.com/s/e1j6bgtg0nyj5im/comp.fst?dl=0)