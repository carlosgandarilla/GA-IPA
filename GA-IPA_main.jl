using DelimitedFiles
using Distances
using Statistics
using PyPlot, PyCall
using SparseArrays
using Random
using LinearAlgebra
using Hungarian
using Bio, FastaIO
using StatsBase
#using CSV
#using DataFrames


include("./dataset_manipulation.jl")
include("./graph_kNN.jl")
include("./graph_Orthologs.jl")
include("./GA_SA.jl")

include("./IPA_main.jl")
include("./compute_MF-DCA_model.jl")
include("./compute_PMI.jl")
include("./predict_pairs.jl")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
GA_IPA_DCA_robust
"""

function GA_IPA_DCA_robust(n_replicates::Int64; kNN = 0::Int64, T_0 =1.0::Float64, alpha = 0.9999::Float64, n_sweep = 40000::Int64, Nincrement = 6::Int64, sim_threshold = 0.3::Float64, pseudocount_weight = 0.5::Float64)

	#n_replicates is the number of realizations of GA experiment, each time taking different random matchings as starting points.
	#kNN is the number of nearest neighbor in the kNN graph. But, if 0 use Orthology.
	#GA parameters
	#T_0 is the initial temperatures of the exponential schedule.
	#alpha is a factor between 0 and 1 to set the temperature T = T_0 * alpha^n_sweep.
	#n_sweep is the number of sweeps, a sweep is defined as N pairing updates.
	#IPA parameters
	#Nincrement is the number of sequences added in each iteration to the training set.
	#sim_threshold is the similarity threshold.
	#pseudocount_weight.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to read the dataset. For example the M = 5052 sequences HK-RR dataset.

	protfile = "./Concat_nnn_extrManySeqs_withFirst.fasta"
	La = 64 #La is the number of aminoacids in the HK protein.
    Lb = 112 #Lb is the number of aminoacids in the RR protein.
	alignA_Num, alignB_Num, dij_A, dij_B, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, N, M = datareader(protfile, La, Lb)
	println("M = ", M)
	println("N = ", N)

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to compute the weights for graph similarity model: kNN or Orthology.

	if kNN == 0#orthology graph (default option)
		wij_A, wij_B = Orthology_Prop(N, M, MSeqSpec, IndexSeqSpec, FirstSeqSpec, LastSeqSpec, dij_A, dij_B)
	else#nearest neighbor graph (kNN graph)
		wij_A, wij_B = kNNprop(kNN, M, dij_A, dij_B)
	end

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to loop over the number of replicates.

	dataTPprotBepi = Array{Float64, 2}(undef, 4 + M, n_replicates)
	#save the number of sequences in this replicate
	dataTPprotBepi[1, :] = fill(M, n_replicates)
	#compute the mean number of pairs per species
	dataTPprotBepi[2, :] = fill(M/N, n_replicates)

	Threads.@threads for k in 1:n_replicates
		dataTPprotBepi[3:4 + M, k] = SA_replicates(T_0, alpha, n_sweep, M, N, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, wij_A, wij_B)
	end

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third.1 step is to save results to file

	if kNN == 0#orthology graph (default option)
		fname = "./HK-RR_GA_Orthology_TP_protB_M=$M._N_sweep=$n_sweep._N_replicates=$n_replicates.text"
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_GA_", kNN, "-NN_TP_protB_M=$M._N_sweep=$n_sweep._N_replicates=$n_replicates.text")
	end

	open(fname, "w") do file
		writedlm(file, dataTPprotBepi, ',')
	end

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to compute robust pairs and put them in a format that IPA-julia can uses.

	id_seqs_robust_pairs_without_singletons, id_seqs_no_robust_pairs_without_singletons = robustness_GA_to_IPA(M, n_replicates, dataTPprotBepi[5:4 + M, :], IndexSeqSpec, MSeqSpec, kNN)

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fifth step is to run DCA-IPA using GA robust pairs as starting set.

	DCA_IPA_GA_robust(kNN, Nincrement, sim_threshold, pseudocount_weight, alignA_Num, alignB_Num, dij_A, dij_B, M, La, Lb, id_seqs_robust_pairs_without_singletons, id_seqs_no_robust_pairs_without_singletons, n_replicates)

end

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

function GA_IPA_MI_robust(n_replicates::Int64; kNN = 0::Int64, T_0 =1.0::Float64, alpha = 0.9999::Float64, n_sweep = 40000::Int64, Nincrement = 6::Int64, sim_threshold = 0.15::Float64, pseudocount_weight = 0.15::Float64)

	#n_replicates is the number of realizations of GA experiment, each time taking different random matchings as starting points.
	#kNN is the number of nearest neighbor in the kNN graph. But, if 0 use Orthology.
	#GA parameters
	#T_0 is the initial temperatures of the exponential schedule.
	#alpha is a factor between 0 and 1 to set the temperature T = T_0 * alpha^n_sweep.
	#n_sweep is the number of sweeps, a sweep is defined as N pairing updates.
	#IPA parameters
	#Nincrement is the number of sequences added in each iteration to the training set.
	#sim_threshold is the similarity threshold.
	#pseudocount_weight.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to read the dataset. For example the M = 5052 sequences HK-RR dataset.

	protfile = "./Concat_nnn_extrManySeqs_withFirst.fasta"
	La = 64 #La is the number of aminoacids in the HK protein.
    Lb = 112 #Lb is the number of aminoacids in the RR protein.
	alignA_Num, alignB_Num, dij_A, dij_B, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, N, M = datareader(protfile, La, Lb)

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to compute the weights for graph similarity model: kNN or Orthology.

	if kNN == 0#orthology graph (default option)
		wij_A, wij_B = Orthology_Prop(N, M, MSeqSpec, IndexSeqSpec, FirstSeqSpec, LastSeqSpec, dij_A, dij_B)
	else#nearest neighbor graph (kNN graph)
		wij_A, wij_B = kNNprop(kNN, M, dij_A, dij_B)
	end

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to loop over the number of replicates.

	dataTPprotBepi = Array{Float64, 2}(undef, 4 + M, n_replicates)
	#save the number of sequences in this replicate
	dataTPprotBepi[1, :] = fill(M, n_replicates)
	#compute the mean number of pairs per species
	dataTPprotBepi[2, :] = fill(M/N, n_replicates)

	Threads.@threads for k in 1:n_replicates

		dataTPprotBepi[3:4 + M, k] = SA_replicates(T_0, alpha, n_sweep, M, N, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, wij_A, wij_B)

	end

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third.1 step is to save results to file

	if kNN == 0#orthology graph (default option)
		fname = "./HK-RR_GA_Orthology_TP_protB_M=$M._N_sweep=$n_sweep._N_replicates=$n_replicates.text"
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_GA_", kNN, "-NN_TP_protB_M=$M._N_sweep=$n_sweep._N_replicates=$n_replicates.text")
	end

	open(fname, "w") do file
		writedlm(file, dataTPprotBepi, ',')
	end

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to compute robust pairs and put them in a format that IPA-julia can uses.

	id_seqs_robust_pairs_without_singletons, id_seqs_no_robust_pairs_without_singletons = robustness_GA_to_IPA(M, n_replicates, dataTPprotBepi[5:4 + M, :], IndexSeqSpec, MSeqSpec, kNN)

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fifth step is to to run MI-IPA using GA robust pairs as starting set.

	MI_IPA_GA_robust(kNN, Nincrement, sim_threshold, pseudocount_weight, alignA_Num, alignB_Num, dij_A, dij_B, M, La, Lb, id_seqs_robust_pairs_without_singletons, id_seqs_no_robust_pairs_without_singletons, n_replicates)

end




#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
robustness_GA_to_IPA returns robust pairs obtained from GA that can be used by IPA as starting set.
"""

function robustness_GA_to_IPA(M::Int64, n_replicates::Int64, TPvsEnergProtB::Array{Float64,2}, IndexSeqSpec::Array{Int64,1}, MSeqSpec::Array{Int64,1}, kNN::Int64)

	#M is the number of sequences.
	#n_replicates is the number of realizations of GA experiment, each time taking different random matchings as starting points.
	#TPvsEnergProtB are the results of GA experiments i.e. matchings returned by GA.
	#IndexSeqSpec is the id of the species of each sequence
	#MSeqSpec is the number of sequences per species.
	#kNN is the number of nearest neighbor in the kNN graph. But, if 0 use Orthology.

	#--------------------------------------------------------------------------------------------------------------------------------------------------
    #The second step is to loop over the sequences in the dataset to find...
	#...how many times a protein A has the same protein B partner.

	repeated_partner = fill(0, M, 3)
	#col 1: ID species
	#col 2: ID sequence protA
	#col 3: ID most repeated partner B
	#repeated_partner contain the most repeated partner, the percent of apearence, ...
	#...if this partner is a correct one and the number of sequences per species.
	ToKeepRobust = Int64[] #id of robust pairs (always repeated pairs).

	for i in 1:M

		#-------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.1 step is find the number of sequences per species to evaluate...
		#...the coomplexity of the task.

		repeated_partner[i, 1] = IndexSeqSpec[i]
		repeated_partner[i, 2] = i

		#-------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.2 step is to find the different protein B partners of a given protein A.

		protAi_partners = TPvsEnergProtB[i, :]
		protAi_unique = unique(protAi_partners)

		#-------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.3 step is to find the different partners, count the number of occurences and returns the higher number.

		(count_protB, ind_protB) = findmax([sum(protAi_partners .== j) for j in protAi_unique])

		#-------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.4 step is to store the most repeated partner.

		repeated_partner[i, 3] = protAi_unique[ind_protB]

		#-------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.5 step is to check if the most repeated partner is the correct partner.

		if count_protB/n_replicates == 1.0
			push!(ToKeepRobust, i)
		end

	end

	#--------------------------------------------------------------------------------------------------------------------------------------------------
    #The third step is to prepare sequences IDs for the singletons suppression.

	id_seqs_translator_without_singletons = fill(0, M, 3)
	#col 1: ID species
	#col 2: ID sequence with singletons
	#col 3: ID sequences after removing singletons

	count_seqs = 0
	ToElim = Int64[] #id of singletons that are going to be eliminated.

	for i in 1:M
		id_seqs_translator_without_singletons[i,1] = IndexSeqSpec[i]
		id_seqs_translator_without_singletons[i,2] = i
		if MSeqSpec[IndexSeqSpec[i]] != 1
			count_seqs += 1
			id_seqs_translator_without_singletons[i,3] = count_seqs
		elseif MSeqSpec[IndexSeqSpec[i]] == 1
			push!(ToElim, i)
		end
	end

	#--------------------------------------------------------------------------------------------------------------------------------------------------
    #The fourth step is to convert robust pairs predicted by GA in something IPA-DCA can use.

	robust_pairs = repeated_partner[setdiff(ToKeepRobust, ToElim), 1:3] #repeated partners without singletons.
	newMseq = size(robust_pairs, 1)

	id_seqs_robust_pairs_without_singletons = fill(0, newMseq, 3)
	#col 1: ID species
	#col 2: ID sequence protA
	#col 3: ID always repeated partner B
	countTP = 0

	for i in 1:newMseq
		id_seqs_robust_pairs_without_singletons[i,1] = robust_pairs[i, 1]
		id_seqs_robust_pairs_without_singletons[i,2] = id_seqs_translator_without_singletons[robust_pairs[i, 2], 3]
		id_seqs_robust_pairs_without_singletons[i,3] = id_seqs_translator_without_singletons[robust_pairs[i, 3], 3]

		if id_seqs_robust_pairs_without_singletons[i,2] == id_seqs_robust_pairs_without_singletons[i,3]
			countTP += 1
		end
	end
	#println("countTP = ", countTP, " newMseq = ", newMseq, " TP = ", countTP/newMseq)

	#--------------------------------------------------------------------------------------------------------------------------------------------------
    #The fifth step is to convert robust pairs predicted by GA in something IPA-DCA can use.

	IDSpec_IDSeq = hcat(IndexSeqSpec, collect(1:M))
	ToKeepPairA = setdiff(collect(1:M), union(ToKeepRobust, ToElim))
	ToKeepPairB = setdiff(collect(1:M), union(repeated_partner[ToKeepRobust, 3], ToElim))
	no_robust_pairs = hcat(IDSpec_IDSeq[ToKeepPairA, :], IDSpec_IDSeq[ToKeepPairB, :])

	id_seqs_no_robust_pairs_without_singletons = fill(0, size(no_robust_pairs, 1), 3)
	#col 1: ID species
	#col 2: ID sequence protA
	#col 3: ID always repeated partner B

	for i in 1:size(no_robust_pairs, 1)
		id_seqs_no_robust_pairs_without_singletons[i,1] = no_robust_pairs[i, 1]
		id_seqs_no_robust_pairs_without_singletons[i,2] = id_seqs_translator_without_singletons[no_robust_pairs[i, 2], 3]
		id_seqs_no_robust_pairs_without_singletons[i,3] = id_seqs_translator_without_singletons[no_robust_pairs[i, 4], 3]
	end

	if length(ToKeepPairA) != length(unique(ToKeepPairB))
		println("No Robust Subset is one-to-one asignment")
	end

	for i = 1:size(no_robust_pairs, 1)
		if no_robust_pairs[i,1] != no_robust_pairs[i,3]
			println("No match inside species for species = ", i)
		end
	end

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The sixth step is to export the...

	if kNN == 0#orthology graph (default option)
		fname = string("./HK-RR_Orthology_robust_pairs_GA_without_singletons_M=$M._N_replicates=$n_replicates.text")
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_", kNN, "-NN_robust_pairs_GA_without_singletons_M=$M._N_replicates=$n_replicates.text")
	end

	open(fname, "w") do file
		writedlm(file, id_seqs_robust_pairs_without_singletons, ',')
	end

	if kNN == 0#orthology graph (default option)
		fname = string("./HK-RR_Orthology-NN_no_robust_pairs_GA_without_singletons_M=$M._N_replicates=$n_replicates.text")
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_", kNN, "-NN_no_robust_pairs_GA_without_singletons_M=$M._N_replicates=$n_replicates.text")
	end

	open(fname, "w") do file
		writedlm(file, id_seqs_no_robust_pairs_without_singletons, ',')
	end

	return id_seqs_robust_pairs_without_singletons, id_seqs_no_robust_pairs_without_singletons

end


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
