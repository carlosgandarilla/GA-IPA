"""
DCA or MI based IPA makes partner prediction using as starting set robust pairs given by GA.
"""

function DCA_IPA_GA_robust(kNN::Int64, Nincrement::Int64, sim_threshold::Float64, pseudocount_weight::Float64, alignA_Num::Array{Int8,2}, alignB_Num::Array{Int8,2}, dij_A::Array{Int64,2}, dij_B::Array{Int64,2}, M::Int64, La::Int64, Lb::Int64, starting_set::Array{Int64,2}, testing_set::Array{Int64,2}, n_replicates::Int64)

	#kNN is the graph ID
	#Nincrement is the number of sequences added in each iteration to the training set.
	#sim_threshold is the similarity threshold.
	#pseudocount_weight.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to find the number of rounds (last one -> all sequences end up in the training set)

	M_start = size(starting_set, 1)
	M_test = size(testing_set, 1)

	Nrounds = ceil(Int64, (M - M_start)./Nincrement + 1)
	println(Nrounds)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to compute the Output matrix: Each row corresponds to an iteration of the DCA-IPA.
	#col 1: number of sequences NSeqs in concatenated alignment used as training set
	#col 2: effective number of sequences Meff in concatenated alignment used as training set
	#col 3: number of TP pairs
	#col 4: number of FP pairs
	#col 5: number of TP pairs in concatenated alignment used as training set
	#col 6: number of FP pairs in concatenated alignment used as training set

	M_new = 0
	Output = zeros(Nrounds, 6)
	Results = zeros(M_test, 6)
	training_set = starting_set

	for rounds = 1:Nrounds #iterate the process until all sequences end up in the training set

		println(rounds)

		if rounds > 1
			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.1 step is to update the training set by adding in the pairings with largest energy gaps made at previous round

			#Use the gap to rank pairs
			for i = 1:M_test
				Results[i, 6] = min(Results[i, 5], Results[i, 6])
			end
			Results = sortslices(Results, dims=1, lt=(x,y)->isless(x[6],y[6]), rev=true)

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.2 step is to number of sequences that will be added to the training set for this round.

			M_new += Nincrement
	        if M_new >= size(Results, 1)
	            M_new  = size(Results, 1) #for the last round, all PAIRED sequences will be in the training set
	        end

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.3 step is to construct new training set with repicking

	        training_set = vcat(starting_set, convert(Array{Int64,2}, Results[1:M_new, 1:3]))

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.4 step is to save to Output the number of TP or FP in the training set

			Output[rounds, 5], Output[rounds, 6] = TP_FP_count(Results, M_new)

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.5 step is to construct model from training set.

	    DCA_couplings_zeroSum, Meff = MFCouplings(sim_threshold, pseudocount_weight, alignA_Num, alignB_Num, dij_A, dij_B, training_set, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.6 step is to test how well the model does by computing scores and pairings in the testing set (INCLUDES SEQS IN TESTING EXCEPT INITIAL GOLD STD)

	    Results = predict_pairs_gap(alignA_Num, alignB_Num, testing_set, M_test, DCA_couplings_zeroSum, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.7 step is to now save the data

		Output[rounds, 1] = M_new + M_start
	    Output[rounds, 2] = Meff
	    Output[rounds, 3], Output[rounds, 4] = TP_FP_count(Results, M_test)
		#

    end

	println("TP = ", Output[end, 3])

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to save the Output matrix and the final pairs made and their scores

	if kNN == 0#orthology graph (default option)
		fname = "./HK-RR_DCA-IPA_TP_deltaE_data_GA_Orthology_robust_pairs_M=$M._N_replicates=$n_replicates._Ninc$Nincrement.txt"
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_DCA-IPA_deltaE_TP_data_GA_", kNN, "-NN_robust_pairs_M=$M._N_replicates=$n_replicates._Ninc$Nincrement.txt")
	end

	open(fname, "w") do file
		writedlm(file, Output, '\t')
	end

	if kNN == 0#orthology graph (default option)
		fname = string("./HK-RR_DCA-IPA_deltaE_Resf_GA_Orthology_robust_pairs_M=$M._N_replicates=$n_replicates._Ninc", Nincrement, "_NStart", M_start, "_round", Nrounds, ".txt")
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_DCA-IPA_deltaE_Resf_GA_", kNN, "-NN_robust_pairs_M=$M._N_replicates=$n_replicates._Ninc", Nincrement, "_NStart", M_start, "_round", Nrounds, ".txt")
	end

	open(fname, "w") do file
		writedlm(file, Results, '\t')
	end

end


#---------------------------------------------------------------------------------------------------------------------------------------------------------------

function MI_IPA_GA_robust(kNN::Int64, Nincrement::Int64, sim_threshold::Float64, pseudocount_weight::Float64, alignA_Num::Array{Int8,2}, alignB_Num::Array{Int8,2}, dij_A::Array{Int64,2}, dij_B::Array{Int64,2}, M::Int64, La::Int64, Lb::Int64, starting_set::Array{Int64,2}, testing_set::Array{Int64,2}, n_replicates::Int64)

	#kNN is the graph ID
	#Nincrement is the number of sequences added in each iteration to the training set.
	#sim_threshold is the similarity threshold.
	#pseudocount_weight is the pseudo-count weight.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to find the number of rounds (last one -> all sequences end up in the training set)

	M_start = size(starting_set, 1)
	M_test = size(testing_set, 1)

	Nrounds = ceil(Int64, (M - M_start)./Nincrement + 1)
	println(Nrounds)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to compute the Output matrix: Each row corresponds to an iteration of the DCA-IPA.
	#col 1: number of sequences NSeqs in concatenated alignment used as training set
	#col 2: effective number of sequences Meff in concatenated alignment used as training set
	#col 3: number of TP pairs
	#col 4: number of FP pairs
	#col 5: number of TP pairs in concatenated alignment used as training set
	#col 6: number of FP pairs in concatenated alignment used as training set

	M_new = 0
	Output = zeros(Nrounds, 6)
	Results = zeros(M_test, 5)
	training_set = starting_set

	for rounds = 1:Nrounds #iterate the process until all sequences end up in the training set

		println(rounds)

		if rounds > 1
			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.1 step is to update the training set by adding in the pairings with largest energy gaps made at previous round

			#Use the gap to rank pairs
			Results = sortslices(Results, dims=1, lt=(x,y)->isless(x[5],y[5]), rev=true)

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.2 step is to number of sequences that will be added to the training set for this round.

			M_new += Nincrement
	        if M_new >= size(Results, 1)
	            M_new  = size(Results, 1) #for the last round, all PAIRED sequences will be in the training set
	        end

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.3 step is to construct new training set with repicking

	        training_set = vcat(starting_set, convert(Array{Int64,2}, Results[1:M_new, 1:3]))

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The second.4 step is to save to Output the number of TP or FP in the training set

			Output[rounds, 5], Output[rounds, 6] = TP_FP_count(Results, M_new)

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.5 step is to construct model from training set.

	    PMIs, Meff = compute_PMIs(sim_threshold, pseudocount_weight, alignA_Num, alignB_Num, dij_A, dij_B, training_set, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.6 step is to test how well the model does by computing scores and pairings in the testing set (INCLUDES SEQS IN TESTING EXCEPT INITIAL GOLD STD)

	    Results = predict_pairs_hungarian_MI(alignA_Num, alignB_Num, testing_set, M_test, PMIs, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.7 step is to now save the data

		Output[rounds, 1] = M_new + M_start
	    Output[rounds, 2] = Meff
	    Output[rounds, 3], Output[rounds, 4] = TP_FP_count(Results, M_test)

    end

	println("TP = ", Output[end, 3])

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to save the Output matrix and the final pairs made and their scores

	if kNN == 0#orthology graph (default option)
		fname = "./HK-RR_MI-IPA_hungE_TP_data_GA_Orthology_robust_pairs._N_replicates=$n_replicates._Ninc$Nincrement.txt"
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_MI-IPA_hungE_TP_data_GA_", kNN, "-NN_robust_pairs_N_replicates=$n_replicates._Ninc$Nincrement.txt")
	end

	open(fname, "w") do file
		writedlm(file, Output, '\t')
	end

	if kNN == 0#orthology graph (default option)
		fname = string("./HK-RR_MI-IPA_hungE_Resf_GA_Orthology_robust_pairs_N_replicates=$n_replicates._Ninc", Nincrement, "_NStart", M_start, "_round", Nrounds, ".txt")
	else#nearest neighbor graph (kNN graph)
		fname = string("./HK-RR_MI-IPA_hungE_Resf_GA_", kNN, "-NN_robust_pairs_N_replicates=$n_replicates._Ninc", Nincrement, "_NStart", M_start, "_round", Nrounds, ".txt")
	end

	open(fname, "w") do file
		writedlm(file, Results, '\t')
	end

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
DCA or MI based IPA makes partner prediction without using a training set as starting set, instead uses a random within species pairings.
"""

function DCA_IPA_no_training(Nincrement::Int64; sim_threshold=0.3::Float64, pseudocount_weight=0.5::Float64, n_rep=1::Int64)

	#Nincrement is the number of sequences added in each iteration to the training set.
	#sim_threshold is the similarity threshold.
	#pseudocount_weight is the pseudo-count weight.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to read the dataset. For example the M = 5052 sequences HK-RR dataset.

	protfile = "./Concat_nnn_extrManySeqs_withFirst.fasta"
	La = 64 #La is the number of aminoacids in the HK protein.
    Lb = 112 #Lb is the number of aminoacids in the RR protein.
	alignA_Num, alignB_Num, dij_A, dij_B, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, N, M = datareader(protfile, La, Lb)

	println(M)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The second step is to start from random within-species pairings: scramble the pairings for this.

	starting_set = ScrambleSeqs(M, N, IndexSeqSpec, FirstSeqSpec, LastSeqSpec)
	testing_set = hcat(hcat(IndexSeqSpec, 1:M), 1:M)

	#number of rounds (last one -> all sequences are in the training set)
	Nrounds = ceil(Int64, M/Nincrement + 1)
	println(Nrounds)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to compute the Output matrix: Each row corresponds to an iteration of the DCA-IPA.
	#col 1: number of sequences NSeqs in concatenated alignment used as training set
	#col 2: effective number of sequences Meff in concatenated alignment used as training set
	#col 3: number of TP pairs
	#col 4: number of FP pairs
	#col 5: number of TP pairs in concatenated alignment used as training set
	#col 6: number of FP pairs in concatenated alignment used as training set

	M_new = 0
	Output = zeros(Nrounds, 6)
	Results = zeros(M, 6)
	training_set = starting_set

	for rounds = 1:Nrounds #iterate the process until all sequences end up in the training set

		println(rounds)

		if rounds > 1
			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.1 step is to update the training set by adding in the pairings with largest energy gaps made at previous round

			#Use the gap to rank pairs
			for i = 1:M
				Results[i, 6] = min(Results[i, 5], Results[i, 6])
			end
			Results = sortslices(Results, dims=1, lt=(x,y)->isless(x[6],y[6]), rev=true)

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.2 step is to number of sequences that will be added to the training set for this round.

			M_new += Nincrement
	        if M_new >= size(Results, 1)
	            M_new  = size(Results, 1) #for the last round, all PAIRED sequences will be in the training set
	        end

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.3 step is to construct new training set

	        training_set = convert(Array{Int64,2}, Results[1:M_new, 1:3])

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.4 step is to save to Output the number of TP or FP in the training set

			Output[rounds, 5], Output[rounds, 6] = TP_FP_count(Results, M_new)

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.5 step is to construct model from training set.

	    DCA_couplings_zeroSum, Meff = MFCouplings(sim_threshold, pseudocount_weight, alignA_Num, alignB_Num, dij_A, dij_B, training_set, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.6 step is to test how well the model does by computing scores and pairings in the testing set (INCLUDES SEQS IN TESTING EXCEPT INITIAL GOLD STD)

	    Results = predict_pairs_gap(alignA_Num, alignB_Num, testing_set, M, DCA_couplings_zeroSum, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.7 step is to now save the data

		Output[rounds, 1] = M_new
	    Output[rounds, 2] = Meff
	    Output[rounds, 3], Output[rounds, 4] = TP_FP_count(Results, M)

    end

	println("TP = ", Output[end, 3])

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to save the Output matrix and the final pairs made and their scores

	fname = string("./HK-RR_DCA-IPA_deltaE_TP_data_no_training_Ninc", Nincrement, "_rep$n_rep.txt")
	open(fname, "w") do file
		writedlm(file, Output, '\t')
	end

	fname = string("./HK-RR_DCA-IPA_deltaE_Resf_no_training_Ninc", Nincrement, "_round", Nrounds, "_rep$n_rep.txt")
	open(fname, "w") do file
		writedlm(file, Results, '\t')
	end

end


#---------------------------------------------------------------------------------------------------------------------------------------------------------------

function MI_IPA_no_training(Nincrement::Int64; sim_threshold=0.15::Float64, pseudocount_weight=0.15::Float64, n_rep=1::Int64)

	#Nincrement is the number of sequences added in each iteration to the training set.
	#sim_threshold is the similarity threshold.
	#pseudocount_weight is the pseudo-count weight..

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to read the dataset. For example the M = 5052 sequences HK-RR dataset.

	protfile = "./Concat_nnn_extrManySeqs_withFirst.fasta"
	La = 64 #La is the number of aminoacids in the HK protein.
    Lb = 112 #Lb is the number of aminoacids in the RR protein.
	alignA_Num, alignB_Num, dij_A, dij_B, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, N, M = datareader(protfile, La, Lb)

	println(M)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The second step is to start from random within-species pairings: scramble the pairings for this.

	starting_set = ScrambleSeqs(M, N, IndexSeqSpec, FirstSeqSpec, LastSeqSpec)
	testing_set = hcat(hcat(IndexSeqSpec, 1:M), 1:M)

	#number of rounds (last one -> all sequences are in the training set)
	Nrounds = ceil(Int64, M/Nincrement + 1)
	println(Nrounds)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to compute the Output matrix: Each row corresponds to an iteration of the DCA-IPA.
	#col 1: number of sequences NSeqs in concatenated alignment used as training set
	#col 2: effective number of sequences Meff in concatenated alignment used as training set
	#col 3: number of TP pairs
	#col 4: number of FP pairs
	#col 5: number of TP pairs in concatenated alignment used as training set
	#col 6: number of FP pairs in concatenated alignment used as training set

	M_new = 0
	Output = zeros(Nrounds, 6)
	Results = zeros(M, 5)
	training_set = starting_set

	for rounds = 1:Nrounds #iterate the process until all sequences end up in the training set

		println(rounds)

		if rounds > 1
			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.1 step is to update the training set by adding in the pairings with largest energy gaps made at previous round

			#Use the gap to rank pairs
			Results = sortslices(Results, dims=1, lt=(x,y)->isless(x[5],y[5]), rev=true)

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.2 step is to number of sequences that will be added to the training set for this round.

			M_new += Nincrement
	        if M_new >= size(Results, 1)
	            M_new  = size(Results, 1) #for the last round, all PAIRED sequences will be in the training set
	        end

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.3 step is to construct new training set with repicking

	        training_set = convert(Array{Int64,2}, Results[1:M_new, 1:3])

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.4 step is to save to Output the number of TP or FP in the training set

			Output[rounds, 5], Output[rounds, 6] = TP_FP_count(Results, M_new)

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.5 step is to construct model from training set.

	    PMIs, Meff = compute_PMIs(sim_threshold, pseudocount_weight, alignA_Num, alignB_Num, dij_A, dij_B, training_set, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.6 step is to test how well the model does by computing scores and pairings in the testing set (INCLUDES SEQS IN TESTING EXCEPT INITIAL GOLD STD)

	    Results = predict_pairs_hungarian_MI(alignA_Num, alignB_Num, testing_set, M, PMIs, La, Lb)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.7 step is to now save the data

		Output[rounds, 1] = M_new
	    Output[rounds, 2] = Meff
	    Output[rounds, 3], Output[rounds, 4] = TP_FP_count(Results, M)

    end

	println("TP = ", Output[end, 3])

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to save the Output matrix and the final pairs made and their scores

	fname = string("./HK-RR_MI-IPA_hungE_TP_data_no_training_Ninc", Nincrement, "_rep$n_rep.txt")
	open(fname, "w") do file
		writedlm(file, Output, '\t')
	end

	fname = string("./HK-RR_MI-IPA_hungE_Resf_no_training_Ninc", Nincrement, "_round", Nrounds, "_rep$n_rep.txt")
	open(fname, "w") do file
		writedlm(file, Results, '\t')
	end

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
counts true and false positive.
"""

function TP_FP_count(Results_matrix::Array{Float64,2}, M::Int64)

	TP_count = 0
	FP_count = 0
	for l = 1:M
		if Results_matrix[l, 2] == Results_matrix[l, 3]
			TP_count += 1
		else
			FP_count += 1
		end
	end

	return TP_count, FP_count

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
returns a random within species pairings.
"""

function ScrambleSeqs(M::Int64, N::Int64, IndexSeqSpec::Array{Int64,1}, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1})

	rand_pairings = hcat(hcat(IndexSeqSpec, 1:M), randmatch(N, FirstSeqSpec, LastSeqSpec))

	return rand_pairings

end
