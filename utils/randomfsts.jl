using ProgressBars


"""
    random_vectorfst(S::Type{<:TensorFSTs.Semiring}, nstates, nisyms, narcs; 
										unweigthed=false,label_offset=1, 
                                        nosyms=nothing, acceptor=false, seed=1234)

TBW
"""
function random_vectorfst(S::Type{<:TensorFSTs.Semiring}, nstates, nisyms, narcs; 
										unweigthed=false, acyclic=false, label_offset=1, nosyms=nothing, acceptor=false,
                                        seed=1234, method="permute")
	if nosyms === nothing
		nosyms = nisyms
	end

	if method == "permute"
		arcs, weights = random_arcs(S, nstates, nisyms, narcs; unweigthed=unweigthed, acyclic=acyclic, nosyms=nosyms, 
									label_offset=label_offset, acceptor=acceptor, seed=seed)
	elseif method=="loop"
		arcs, weights = random_arcs_loop(S, nstates, nisyms, narcs; unweigthed=unweigthed, acyclic=acyclic, nosyms=nosyms, 
									label_offset=label_offset, acceptor=acceptor, seed=seed)
	end
	
	A = TensorFSTs.Arc{S}
	t = Vector{TensorFSTs.Arc{S}}
    tarcs = Vector{t}()
    for i in 1:nstates
        push!(tarcs,Vector{t}())
    end

    for (a,w) in zip(arcs, weights)
		push!(tarcs[a[3]], A(a[1],a[2],w,a[4]))
    end
	final = zeros(S,nstates)
	if unweigthed
		final[nstates] = one(S)
	else
		final[nstates] = S(rand())
	end
    VectorFST(tarcs, 1, final)
end

function random_arcs_loop(S::Type{<:TensorFSTs.Semiring}, nstates, nisyms, narcs; unweigthed=false, acyclic=false,
                        label_offset=1, nosyms=nothing, acceptor=false, seed=1234)

	if nosyms === nothing
		nosyms = nisyms
	end	

	Random.seed!(seed)
	arcs = []
	weights = []

	if acceptor
		total_arcs = nstates^2*(nisyms-label_offset+1)
		if narcs > total_arcs
			throw(ArgumentError("Number of arcs must be less than nstates^2 nsyms, $(narcs) > $(total_arcs)"))
		end
	else
		total_arcs = nstates^2*(nisyms-label_offset+1)*(nosyms-label_offset+1)
		if narcs > total_arcs
			throw(ArgumentError("Number of arcs must be less than nstates^2 nisyms nosyms, $(narcs) > $(total_arcs) "))
		end
	end
	# println("Narcs: ", narcs)

	keys = Dict()

	for n in 1:narcs
		notok = true
		while notok

			if acceptor
				il = rand(label_offset:nisyms-label_offset+1)
				ol = il
			else
				il,ol = rand(label_offset:nisyms-label_offset+1), rand(label_offset:nisyms-label_offset+1)
			end

			if acyclic
				ss = rand(1:nstates-1)
				ds = rand(ss+1:nstates)
			else
				ss,ds = rand(1:nstates), rand(1:nstates)				
			end

			notok = false

			arc = (il, ol, ss, ds)

			if haskey(keys, arc)
				notok = true
			end

			if !notok
				keys[arc] = true
				push!(arcs, arc)
				if unweigthed
					push!(weights, one(S))
				else
					push!(weights, S(rand()))
				end
			end
		end	
	end
	p = sortperm(arcs)
	arcs = arcs[p]
	weights = weights[p]
	arcs, weights
end

function random_arcs(S::Type{<:TensorFSTs.Semiring}, nstates, nisyms, narcs; unweigthed=false, acyclic=false,
                        label_offset=1, nosyms=nothing, acceptor=false, seed=1234)

	if nosyms === nothing
		nosyms = nisyms
	end	

	Random.seed!(seed)
	arcs = []
	weights = []

	if acceptor
		total_arcs = nstates^2*(nisyms-label_offset+1)
		if narcs > total_arcs
			throw(ArgumentError("Number of arcs must be less than nstates^2 nsyms, $(narcs) > $(total_arcs)"))
		end
		cis = CartesianIndices((nstates,nstates,(nisyms-label_offset+1)))
	else
		total_arcs = nstates^2*(nisyms-label_offset+1)*(nosyms-label_offset+1)
		if narcs > total_arcs
			throw(ArgumentError("Number of arcs must be less than nstates^2 nisyms nosyms, $(narcs) > $(total_arcs) "))
		end
		cis = CartesianIndices((nstates,nstates,(nisyms-label_offset+1),(nosyms-label_offset+1)))
	end


	if acyclic
		new_cis = []
		for c in cis
			if c[1]<c[2]
				push!(new_cis,c)
			end
		end
		cis = new_cis

		if acceptor
			total_arcs = floor(Int,(nstates^2-nstates)/2*(nisyms-label_offset+1))
			if narcs > total_arcs
				throw(ArgumentError("Number of arcs must be less than nstates(nstates-1)/2 nsyms, $(narcs) > $(total_arcs)"))
			end
		else
			total_arcs = floor(Int,(nstates^2-nstates)/2*(nisyms-label_offset+1)*(nosyms-label_offset+1))
			if narcs > total_arcs
				throw(ArgumentError("Number of arcs must be less than  nstates(nstates-1)/2 nisyms nosyms, $(narcs) > $(total_arcs) "))
			end
		end
	end

	shuffled_arcs = shuffle(1:total_arcs)

	for arc_ix in shuffled_arcs[1:narcs]
		if acceptor
			ss,ds,il = Tuple(cis[arc_ix])
			ol = il
		else
			ss,ds,il,ol = Tuple(cis[arc_ix])
		end

		il = il + label_offset - 1
		ol = ol + label_offset - 1
		
		arc = (il,ol, ss, ds)
		push!(arcs, arc)
		if unweigthed
			push!(weights, one(S))
		else
			push!(weights, S(rand()))
		end
	end
	p = sortperm(arcs)
	arcs = arcs[p]
	weights = weights[p]
	arcs, weights
end