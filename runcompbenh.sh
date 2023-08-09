# JULIA

JULIAPROJPATH="../wfstproj"
FSADBPATH="fsadb_uw"

cd ../TensorFSTs.jl && git checkout coo_composition && cd ../FSM-Ops-Benchmarking
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH TensorFSTs_onlycomp
julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH OpenFst
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CscOfCooLod
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooLod
# julia  --threads 8 --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooMTLod


cd ../TensorFSTs.jl && git checkout refactor && cd ../FSM-Ops-Benchmarking
julia  --project="../newwfstproj"  composition/composition_benchmark.jl $FSADBPATH FiberTensorFSTs

# K2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python composition/composition_benchmark_k2.py cpu $FSADBPATH --other_db results/fsadb_uw_OpenFst_compbenchs.csv
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=8 python composition/composition_benchmark_k2.py cuda $FSADBPATH --other_db results/fsadb_uw_OpenFst_compbenchs.csv
# OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python composition/composition_benchmark_k2.py cpu $FSADBPATH


# # JULIA

# JULIAPROJPATH="../wfstproj"
# # FSADBPATH="fsadb_uw"
# FSADBPATH="fstdb_tropical_2"

# # julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooNoallocLod
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfDictLod &
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooSod &
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CscOfCooLod &
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooLod &
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH OpenFst

# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH TensorFSTs &
# julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH TensorFSTs_onlycomp &
# julia  --threads 8 --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooMTLod &



# # K2

# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python composition/composition_benchmark_k2.py cpu $FSADBPATH

# OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python composition/composition_benchmark_k2.py cpu $FSADBPATH

# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=8 python composition/composition_benchmark_k2.py cuda $FSADBPATH