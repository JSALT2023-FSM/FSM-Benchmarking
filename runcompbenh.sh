# JULIA

JULIAPROJPATH="../wfstproj"
FSADBPATH="fsadb_uw"

julia  --threads 4 --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCooMT
julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CscOfCoo
julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH CooOfCoo
julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH OpenFst
julia  --project=$JULIAPROJPATH  composition/composition_benchmark.jl $FSADBPATH TensorFSTs

# K2

OMP_NUM_THREADS=1 python composition/composition_benchmark_k2.py
cp fsadb_uw_composed_k2_compbenchs.csv fsadb_uw_composed_k2_compbenchs_th1.csv

OMP_NUM_THREADS=4 python composition/composition_benchmark_k2.py
cp fsadb_uw_composed_k2_compbenchs.csv fsadb_uw_composed_k2_compbenchs_th4.csv