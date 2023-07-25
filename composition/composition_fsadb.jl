"""
Gets all combinations of two FSTs from a directory and composes them to a new directory. 
Creates a CSV file with the metadata of the composition (nstates and narcs).
"""

# Directory to look for FSTs in binary format
dbname = ARGS[1]
total_composition = 4000
minarcs = 12
maxarcs = 50000

using DataFrames
import IterTools.product
using ProgressBars
using CSV
using OpenFst
using TensorFSTs
using Random

Random.seed!(123456)

OF=OpenFst
TF=TensorFSTs
SR=TF.Semirings

dbname_composed = "data/" * dbname * "_composed"

if !isdir(dbname_composed)
    mkdir(dbname_composed)
end

root, dirs, file = first(walkdir("data/" * dbname))

# Filter out FSTs with less than minarcs and more than maxarcs
narcs = map(x->parse(Int,split(split(x,"-")[2],"_")[2]),file)
file = file[minarcs.>12 .&& narcs.<maxarcs]
files = joinpath.(root, file)

products = vec(collect(product(files, files)))
products = shuffle(products)

@show "Number of combinations of FSTs: $(length(products))"

lk = ReentrantLock()

results = []
Threads.@threads for i in ProgressBar(1:total_composition)
    f1,f2 = products[i]
    n1 = split(basename(f1), "_")[1]
    n2 = split(basename(f2), "_")[1]

    outputname = joinpath( dbname_composed, "$(n1)_$(n2).fst")

    A = OF.read(f1)
    B = OF.read(f2)
    C = OF.compose(A,B)
    
    nstates = OF.numstates(C)
    if nstates>0
        narcs = sum([OF.numarcs(C,i) for i in 1:OF.numstates(C)])
    else
        narcs = 0
    end

    lock(lk) do
        if nstates>0
            OF.write(C, outputname)
            push!(results, (fileA=f1, fileB=f2, fileC=outputname,  nstates=nstates, narcs=narcs) )
        else
            push!(results, (fileA=f1, fileB=f2, fileC="",  nstates=nstates, narcs=narcs) )
        end    
        CSV.write("data/$(dbname_composed).csv", DataFrame(results))
    end    
end
