using DataFrames
import IterTools.product
using ProgressBars
using CSV
using OpenFst
using TensorFSTs
using BenchmarkTools
using Statistics
using ProgressMeter

OF = OpenFst
TF = TensorFSTs
SR = TF.Semirings

# Include a sign to applied to the OpenFst floating point weight
_SemiringToWeightType = Dict([
    (SR.LogSemiring{Float32,1}, (OF.LogWeight, -1))
    (SR.LogSemiring{Float32,-1}, (OF.LogWeight, 1))
    (SR.LogSemiring{Float64,1}, (OF.Log64Weight, -1))
    (SR.LogSemiring{Float64,-1}, (OF.Log64Weight, 1))
    (SR.LogSemiring{Float32,Inf}, (OF.TropicalWeight, -1))
    (SR.LogSemiring{Float32,-Inf}, (OF.TropicalWeight, 1))
])

# Converts from OpenFst weight to TensorFSTs semiring
_WeightToSemiringType = Dict([
    (OF.LogWeight, SR.LogSemiring{Float32,-1}),
    (OF.Log64Weight, SR.LogSemiring{Float64,-1}),
    (OF.TropicalWeight, SR.LogSemiring{Float32,-Inf})
])

# Extracts semiring floating point value with sgn correction
function _semiring_to_weight(s::S, sgn)::AbstractFloat where {S<:SR.Semiring}
    s.val * sgn
end

function OF.VectorFst(tfst::TF.ExpandedFST{S}) where {S<:SR.Semiring}
    W, sgn = _SemiringToWeightType[S]
    ofst = OF.VectorFst{W}()
    # We need expanded for this line only
    OF.reservestates(ofst, numstates(tfst))
    for s in states(tfst)
        OF.addstate!(ofst)
        final = _semiring_to_weight(TF.final(tfst, s), sgn)
        OF.setfinal!(ofst, s, final)
        OF.reservearcs(ofst, s, numarcs(tfst, s))
        for a in arcs(tfst, s)
            arc = OF.Arc(ilabel=a.ilabel,
                olabel=a.olabel,
                weight=_semiring_to_weight(a.weight, sgn),
                nextstate=a.nextstate)
            OF.addarc!(ofst, s, arc)
        end
    end
    OF.setstart!(ofst, start(tfst))
    return ofst
end

function TF.VectorFST(ofst::OF.Fst{W}) where {W<:OF.Weight}
    S = _WeightToSemiringType[W]
    tfst = TF.VectorFST{S}()
    for s in OF.states(ofst)
        TF.addstate!(tfst)
        final = S(OF.final(ofst, s))
        TF.setfinal!(tfst, s, final)
        for a in OF.arcs(ofst, s)
            arc = TF.Arc(Int(a.ilabel), Int(a.olabel), S(a.weight),
                Int(a.nextstate))
            TF.addarc!(tfst, s, arc)
        end
    end
    TF.setstart!(tfst, OF.start(ofst))
    return tfst
end


S = SR.TropicalSemiring{Float32}


function OF.numarcs(A::OF.VectorFst)
    sum([OF.numarcs(A, i) for i in 1:OF.numstates(A)])
end

function TF.numarcs(A::TF.VectorFST)
    sum([TF.numarcs(A, i) for i in 1:TF.numstates(A)])
end

function bench(A, B, compose_fn)
    b = @benchmarkable $compose_fn($A, $B)
    # tune!(b)
    t = run(b, samples=1000, seconds=0.1, evals=1)
    t.times
end

function compose_check_and_bench(A, B, C, preprocess_fn, compose_fn; postprocess_fn=nothing)
   
    # println("Preprocessing...\t")
    A = preprocess_fn(A)
    B = preprocess_fn(B)

    # println("Composing...\t")
    D = compose_fn(A, B)
    # print(typeof(D))

    if postprocess_fn!==nothing
        D = postprocess_fn(D)
    end 

    # println("Convert...\t")
    if isa(D, OF.VectorFst)
        # D = OF.VectorFst(D)
        D_nstates = OF.numstates(D)
        D_narcs = OF.numarcs(D)
        F_nstates = NaN
        F_narcs = NaN
    elseif isa(D, TF.VectorFST)
        D_nstates = TF.numstates(D)
        D_narcs = TF.numarcs(D)
        F_nstates = NaN
        F_narcs = NaN
    elseif isa(D, TF.TensorFST)
        D_nstates = TF.numstates(D)
        D_narcs = TF.numarcs(D)
        F_nstates = NaN
        F_narcs = NaN
    end
    
    # println("nstates...\t")
    # D_nstates = OF.numstates(D)

    # println("narcs...\t")
    # D_narcs = OF.numarcs(D)

    # println("Connect...\t")
    # F = OF.connect(D)
    # F_nstates = OF.numstates(F)
    # F_narcs = OF.numarcs(F)

    # println("Connect...\t")
    # C = OF.connect(C)


    # print("Benchmarking...\t")
    times = bench(A, B, compose_fn)

    # print("Checking equivalence...\t")    
    # equivalence = OF.equivalent(OF.determinize(C), OF.determinize(F))
    equivalence = NaN

    times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence
end

preprocess_CooOfCoo(A) = vectorFst2COOofCOOFst(A)
compose_CooOfCoo(A, B) = COOofCOO_compose(A, B, "cscsum_alloc")
compose_CooOfCooMT(A, B) = COOofCOO_compose(A, B "cscsum_mt")
postprocess_CooOfCoo(C) = TF.VectorFST(coo_lod2arcs(C["coo"], C["Q"], C["S"]), 1, C["finalweights"])

preprocess_tensorfst(A) = convert(TF.TensorFST{S}, TF.VectorFST(A))
compose_tensorfst(A, B) = TF.compose(A, B)

compose_openfst(A, B) = OF.compose(A, B, false)

function named_tuple_maker(r, times, name, nstates, narcs, conn_nstates, conn_narcs, equivalence	)
    (
        fileA=r["fileA"], fileB=r["fileB"], fileC=r["fileC"],
        min=minimum(times), max=maximum(times), mean=mean(times), std=std(times),
        name=name, nstates=nstates, narcs=narcs, conn_nstates=conn_nstates, conn_narcs=conn_narcs,
        equivalence=equivalence
    )
end

dbname = "fsadb_uw"
dbname_composed = dbname * "_composed"

df = CSV.read("$(dbname_composed).csv", DataFrame)
df = df[df.nstates.>0,:]


# lk = ReentrantLock()
mode = ARGS[1]

outputname = "$(dbname)_$(mode)_compbenchs.csv"

if isfile(outputname)
    dfpre = CSV.read(outputname, DataFrame)
else
    dfpre = nothing
end

results = []
for i in ProgressBar(1:size(df,1))
# for i in ProgressBar(1:10)
    r = df[i,:]
    A = OF.read(r["fileA"])
    B = OF.read(r["fileB"])
    C = OF.read(String(r["fileC"]))

    # check dfpre is not nothing and if the composition has already been done

    if dfpre!==nothing && String(r["fileC"]) in dfpre[!,"fileC"]
        continue
    end
    
    # @show OF.numstates(A), OF.numstates(B), OF.numstates(C), OF.numarcs(A), OF.numarcs(B), OF.numarcs(C)

    if OF.numarcs(C)>500000
        continue
    end

    if mode=="OpenFst"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, identity, compose_openfst)
        push!(results,named_tuple_maker(r, times,  "OpenFst", D_nstates, D_narcs, F_nstates, F_narcs, equivalence))
    end

    if mode=="CooOfCoo"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCoo, compose_CooOfCoo; postprocess_fn=postprocess_CooOfCoo)   
        push!(results,named_tuple_maker(r, times,  "CooOfCoo", D_nstates, D_narcs, F_nstates, F_narcs, equivalence))
    end

    if mode=="CooOfCooMT"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCoo, compose_CooOfCooMT; postprocess_fn=postprocess_CooOfCoo)   
        push!(results,named_tuple_maker(r, times,  "CooOfCoo", D_nstates, D_narcs, F_nstates, F_narcs, equivalence))
    end


    if mode=="TensorFSTs"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_tensorfst, compose_tensorfst)
        push!(results,named_tuple_maker(r, times,  "TensorFSTs", D_nstates, D_narcs, F_nstates, F_narcs, equivalence))   
    end

    CSV.write(outputname, DataFrame(results))   
end

