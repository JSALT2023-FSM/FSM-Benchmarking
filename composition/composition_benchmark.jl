using DataFrames
import IterTools.product
using ProgressBars
using CSV
using OpenFst
using TensorFSTs
using BenchmarkTools
using Statistics
using Random
using NPZ


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

# function OF.VectorFst(tfst::TF.ExpandedFST{S}) where {S<:SR.Semiring}
#     W, sgn = _SemiringToWeightType[S]
#     ofst = OF.VectorFst{W}()
#     # We need expanded for this line only
#     OF.reservestates(ofst, numstates(tfst))
#     for s in states(tfst)
#         OF.addstate!(ofst)
#         final = _semiring_to_weight(TF.final(tfst, s), sgn)
#         OF.setfinal!(ofst, s, final)
#         OF.reservearcs(ofst, s, numarcs(tfst, s))
#         for a in arcs(tfst, s)
#             arc = OF.Arc(ilabel=a.ilabel,
#                 olabel=a.olabel,
#                 weight=_semiring_to_weight(a.weight, sgn),
#                 nextstate=a.nextstate)
#             OF.addarc!(ofst, s, arc)
#         end
#     end
#     OF.setstart!(ofst, start(tfst))
#     return ofst
# end

function TF.TensorFST(ofst::OF.Fst{W}) where W <: OF.Weight
    S = _WeightToSemiringType[W] 
    arcs = []
    finals = []
    for s in OF.states(ofst)
        push!(finals, s=>S(OF.final(ofst, s)) )
        for a in OF.arcs(ofst, s)
            arc = (src=Int(s), isym=Int(a.ilabel), osym=Int(a.olabel), weight=S(a.weight), dest=Int(a.nextstate))
            push!(arcs, arc)
        end
    end
    tfst = TF.TensorFST(
    S,
    arcs,
    [OF.start(ofst) => one(S)],
    finals
)
    return tfst
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

function bench(A, B, compose_fn, seconds)
    b = @benchmarkable $compose_fn($A, $B; bench=true)
    # tune!(b)
    t = run(b, samples=500, seconds=seconds, evals=1)
    t.times
end

function numlabels(A::OF.VectorFst)
    iL = 1
    oL = 1
    for s in OF.states(A)
        for a in OF.arcs(A, s)
            iL = max(iL, a.ilabel)
            oL = max(oL, a.olabel)
        end
    end
    iL, oL
end

# TF.VectorFST = Nothing

function compose_check_and_bench(A, B, C, preprocess_fn, compose_fn; postprocess_fn=nothing, seconds=0.2)
   
    println("Preprocessing...\t")
    iA, oA = numlabels(A)
    iB, oB = numlabels(B)
    println("Number of states:\t", OF.numstates(A), "\t", OF.numstates(B), "\t", OF.numstates(C))
    println("Number of arcs:\t", OF.numarcs(A), "\t", OF.numarcs(B), "\t", OF.numarcs(C))
    println("Number of labels:\t", iA, "\t", oA, "\t", iB, "\t", oB)    

    A = preprocess_fn(A, iA, max(oA, iB), "A")
    B = preprocess_fn(B, max(oA, iB), oB, "B")


    println("Composing...\t")
    D = compose_fn(A, B)
    # print(typeof(D))

    
    if postprocess_fn!==nothing
        D = postprocess_fn(A,B,D)
    end 

    println("Convert...\t")
    if isa(D, OF.VectorFst)
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
    else
        error("Unknown type")
    end
    
    equivalence = NaN
    F_nstates = NaN
    F_narcs = NaN

    # check equivalence only if the number of states is small
    # if D_narcs<100
    #     if isa(D, TF.VectorFST) || isa(D, TF.TensorFST)
    #         F = OF.connect(OF.VectorFst(D))
    #         F_nstates = OF.numstates(F)
    #         F_narcs = OF.numarcs(F)
    #         print("Checking equivalence...\t")
    #         equivalence = OF.equivalent(OF.determinize(F), OF.determinize(C))
    #     end
    # end

    print("Benchmarking...\t")
    times = bench(A, B, compose_fn, seconds)

    times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence
end

preprocess_CooOfCooLod(A, nisym, nosym, x) = vectorFst2COOofCOOFst(TF.VectorFST(A),  nisym, nosym, "lod")
compose_CooOfCooLod(A, B; bench=false) = COOofCOO_compose(A, B, "lod_cscsum_alloc")
compose_CooOfCooNoallocLod(A, B; bench=false) = COOofCOO_compose(A, B, "lod_cscsum_noalloc")
compose_CscOfCooLod(A, B; bench=false) = COOofCOO_compose(A, B, "lod_cscsum_cscmul")
compose_CooOfCooMTLod(A, B; bench=false) = COOofCOO_compose(A, B, "lod_cscsum_mt")
postprocess_CooOfCooLod(A,B,C) = TF.VectorFST(coo_lod2arcs(C["coo"], C["Q"], C["S"]), 1, C["finalweights"])

preprocess_CooOfDictLod(A, nisym, nosym, x) = vectorFst2COOofCOOFst(TF.VectorFST(A),  nisym, nosym, "lod_dict")
compose_CooOfDictLod(A, B; bench=false) = COOofCOO_compose(A, B, "lod_dictsum")
postprocess_CooOfDictLod(A,B,C) = TF.VectorFST(cooofdict_lod2arcs(C["coo"], C["Q"], C["S"]), 1, C["finalweights"])

preprocess_CooOfCooSod(A, nisym, nosym, x) = vectorFst2COOofCOOFst(TF.VectorFST(A),  nisym, nosym, "sod")
compose_CooOfCooSod(A, B; bench=false) = COOofCOO_compose(A, B, "sod_kroncoo_cscmul")
postprocess_CooOfCooSod(A,B,C) = TF.VectorFST(coo_sod2arcs(C["coo"], C["S"]), 1, C["finalweights"])

preprocess_tensorfst(A, nisym, nosym, x) = convert(TF.TensorFST{S}, TF.VectorFST(A))
compose_tensorfst(A, B; bench=false) = TF.compose(A, B)

function preprocess_fibertensorfst(A, nisym, nosym, x)
    if x=="A"
        A = sort(reorient(TF.TensorFST(A), (:olabel, :src, :dest, :ilabel)))
    elseif x=="B"
    	A =sort(reorient(TF.TensorFST(A), (:ilabel, :src, :dest, :olabel)))
    end    
end
compose_fibertensorfst(A, B; bench=false) = TF.compose(A, B)

preprocess_fibertensorfstReorient(A, nisym, nosym, x) = TF.TensorFST(A)
compose_fibertensorfstReorient(A, B; bench=false) = TF.compose(sort(reorient(A, (:olabel, :src, :dest, :ilabel))),sort(reorient(B, (:ilabel, :src, :dest, :olabel))))


function preprocess_tensorfst_onlycomp(A, nisym, nosym, x)
    A = convert(TF.TensorFST{S}, TF.VectorFST(A))
    if x=="A"
        A = reorder(A, (:olabel, :ilabel, :src, :dest))
    elseif x=="B"
        A = reorder(A, (:ilabel, :olabel, :src, :dest))
    end
    return A
end

compose_tensorfst_onlycomp(A, B; bench=false) = TF.sumtensorproduct2(A.M, B.M, bench)
function postprocess_tensorfst_onlycomp(A, B, C)
    M = TF.sparse_csr( C[1],C[2],C[3],C[4],C[5]; dims=C[6] )
    # M = sparse_csr(newI[nzind], newJ[nzind], newK[nzind], newL[nzind], newV[nzind]; dims)
    TF.TensorFST{S,(:ilabel, :olabel, :src, :dest)}(
        M,
        start(B) + (start(A) - 1) * numstates(B),
        kron(A.ω, B.ω)
    )
end 


preprocess_cukron(A, nisym, nosym, x) = TF.vector2cuCoo(TF.VectorFST(A))

compose_cukron(A, B; bench=false) = TF.cuFSAComp(A, B)
function postprocess_cukron(A, B, C)
    TF.VectorFST(TF.coo2arcs(C[1],C[2],C[3],C[4],C[5], A["Q"]*B["Q"]), 1, TF.kron(A["finalweights"], B["finalweights"]))
end


preprocess_openfst(A, nisym, nosym, x) = A
compose_openfst(A, B; bench=false) = OF.compose(A, B, false)

function named_tuple_maker(r, times, name, nstates, narcs, conn_nstates, conn_narcs, equivalence	)
    (
        fileA=r["fileA"], fileB=r["fileB"], fileC=r["fileC"],
        min=minimum(times), max=maximum(times), mean=mean(times), std=std(times),
        name=name, nstates=nstates, narcs=narcs, conn_nstates=conn_nstates, conn_narcs=conn_narcs,
        equivalence=equivalence
    )
end

dbname = ARGS[1]
# dbname = "fsadb_uw"

dbname_composed = "data/$(dbname)_composed.csv"
@show dbname_composed 

df = CSV.read(dbname_composed, DataFrame)
df = df[df.nstates.>0,:]

# lk = ReentrantLock()
mode = ARGS[2]
# mode = "CooOfCooSod"
@show mode 

outputname = "results/$(dbname)_$(mode)_compbenchs.csv"
@show outputname

if isfile(outputname)
    dfpre = CSV.read(outputname, DataFrame)
    #DataFrame to list of named tuples
    # results = [named_tuple_maker(r, r["times"], r["name"], r["nstates"], r["narcs"], r["conn_nstates"], r["conn_narcs"], r["equivalence"]) for r in eachrow(dfpre)]
else
    dfpre = nothing
end

results = []

Random.seed!(123456)
ids = shuffle(2:size(df,1))[1:1000]
ids = pushfirst!(ids, 1)
println(ids[1:10])

# for i in ProgressBar(1:size(df,1))
for i in ProgressBar(ids)
    r = df[i,:]

    if !isfile(strip(r["fileA"])) || !isfile(strip(r["fileB"])) || !isfile(strip(String(r["fileC"])))
        println("File not found ", r["fileA"]," ", r["fileB"]," ", r["fileC"])
        continue
    end

    A = OF.read(r["fileA"])
    B = OF.read(r["fileB"])
    C = OF.read(String(r["fileC"]))

    # check dfpre is not nothing and if the composition has already been done

    # if dfpre!==nothing && String(r["fileC"]) in dfpre[!,"fileC"]
    #     continue
    # end
    
    # @show OF.numstates(A), OF.numstates(B), OF.numstates(C), OF.numarcs(A), OF.numarcs(B), OF.numarcs(C)

    # if OF.numarcs(C)>500000
    #     continue
    # end

    # if OF.numarcs(C)>50000 || OF.numarcs(A)>50000 || OF.numarcs(B)>50000
    #     continue
    # end

    if i==1
        seconds = 60
    else
        seconds = 0.2
    end

    if mode=="CuKron"
        try    
            times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_cukron, compose_cukron; postprocess_fn= postprocess_cukron, seconds=seconds)
        catch
            println("Skipping ", r["fileA"]," ", r["fileB"]," ", r["fileC"])
            times = [-1]
            D_nstates = NaN
            D_narcs = NaN
            F_nstates = NaN
            F_narcs = NaN
            equivalence = NaN
        end
    end
    

    if mode=="FiberTensorFSTs"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_fibertensorfst, compose_fibertensorfst; seconds=seconds)
    end
    
    if mode=="FiberTensorFSTsReorient"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_fibertensorfstReorient, compose_fibertensorfstReorient; seconds=seconds)
    end

    if mode=="TensorFSTs_onlycomp"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_tensorfst_onlycomp, compose_tensorfst_onlycomp; postprocess_fn=postprocess_tensorfst_onlycom, seconds=seconds)
    end

    if mode=="TensorFSTs"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_tensorfst, compose_tensorfst; seconds=seconds)
    end

    if mode=="CooOfDictLod"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfDictLod, compose_CooOfDictLod; postprocess_fn=postprocess_CooOfDictLod, seconds=seconds)   
    end

    if mode=="CooOfCooSod"
        if r["fileA"]=="data/real/ashrafsamplesent.noeps.fst"
            continue
        end
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCooSod, compose_CooOfCooSod; postprocess_fn=postprocess_CooOfCooSod, seconds=seconds)   
    end

    if mode=="OpenFst"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_openfst, compose_openfst; seconds=seconds)
    end

    if mode=="CooOfCooNoallocLod"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCooLod, compose_CooOfCooNoallocLod; postprocess_fn=postprocess_CooOfCooLod, seconds=seconds)   
    end

    if mode=="CooOfCooLod"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCooLod, compose_CooOfCooLod; postprocess_fn=postprocess_CooOfCooLod, seconds=seconds)   
    end

    if mode=="CooOfCooMTLod"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCooLod, compose_CooOfCooMTLod; postprocess_fn=postprocess_CooOfCooLod, seconds=seconds)   
    end

    if mode=="CscOfCooLod"
        times, D_nstates, D_narcs, F_nstates, F_narcs, equivalence = compose_check_and_bench(A, B, C, preprocess_CooOfCooLod, compose_CscOfCooLod; postprocess_fn=postprocess_CooOfCooLod, seconds=seconds)   
    end
    

    if i==1
        npzwrite("results/realexample_times_$(mode).npy", times)
    end

    push!(results, named_tuple_maker(r, times,  mode, D_nstates, D_narcs, F_nstates, F_narcs, equivalence))   

    CSV.write(outputname, DataFrame(results))   
end

