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