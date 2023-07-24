julia  --threads 4 --project="../wfstproj"  composition/composition_benchmark.jl CooOfCooMT
julia  --project="../wfstproj"  composition/composition_benchmark.jl CscOfCoo
julia  --project="../wfstproj"  composition/composition_benchmark.jl CooOfCoo
julia  --project="../wfstproj"  composition/composition_benchmark.jl OpenFst
julia  --project="../wfstproj"  composition/composition_benchmark.jl TensorFSTs

OMP_NUM_THREADS=1 python composition/composition_benchmark_k2.py
cp fsadb_uw_composed_k2_compbenchs.csv fsadb_uw_composed_k2_compbenchs_th1.csv
OMP_NUM_THREADS=4 python composition/composition_benchmark_k2.py
cp fsadb_uw_composed_k2_compbenchs.csv fsadb_uw_composed_k2_compbenchs_th4.csv