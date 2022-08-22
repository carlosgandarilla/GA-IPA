#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
"""
calculate PMI-based pairing scores between all pairs of As and Bs in test_seqs
"""

function compute_pairing_scores_MI(alignA_Num::Array{Int8,2}, alignB_Num::Array{Int8,2}, test_seqs::Array{Int64,2}, M_test::Int64, PMIs::Array{Float64,4}, La::Int64, Lb::Int64)

	#alignA_Num contains alignment A after the aminoacids being converted into numbers, on which the model is trained.
	#alignB_Num contains alignment B after the aminoacids being converted into numbers, on which the model is trained.
	#test_seqs is an array containing the id of sequences in an species, i.e. id of seq A and id of seq B.
	#M_test is the number of sequences in a "test_seqs" (subset of the testing set). .
    #PMIs is the mutual information for joint A-B alignment.
	#La is the number of aminoacids in the alignment A.
	#Lb is the number of aminoacids in the alignment B.

    pairing_scores = zeros(M_test, M_test)

    for i = 1:M_test #to choose the A
        for j = 1:M_test #to choose the B
            for a = 1:La #sites in A
                for b = 1:Lb #sites in B
                    #nb here a < b always, so it is fine to just store half of Wstore.
                    aa1 = alignA_Num[test_seqs[i, 1], a] #aa in A i at site a
                    aa2 = alignB_Num[test_seqs[j, 2], b] #aa in B j at site b
                    pairing_scores[i, j] += PMIs[a, La + b, aa1, aa2]
                end
            end
        end
    end

    return pairing_scores

end



#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
"""
compute_PMIs returns the Mutual Information (MI) couplings of the concatenated MSA.
"""

function compute_PMIs(sim_threshold::Float64, pseudocount_weight::Float64, alignA_Num::Array{Int8,2}, alignB_Num::Array{Int8,2}, dij_A::Array{Int64,2}, dij_B::Array{Int64,2}, training_set::Array{Int64,2}, La::Int64, Lb::Int64)

	#sim_threshold is the similarity threshold.
	#pseudocount_weight is the pseudo-count weight
	#alignA_Num contains alignment A after the aminoacids being converted into numbers, on which the model is trained.
	#alignB_Num contains alignment B after the aminoacids being converted into numbers, on which the model is trained.
	#dij_A is the Hamming distance between sequences in the alignment A.
	#dij_B is the Hamming distance between sequences in the alignment B.
	#training_set is an array containing the id of species, id of seq A and id of seq B.
	#La is the number of aminoacids in the alignment B.
	#Lb is the number of aminoacids in the alignment A.
	q ::Int8 = 21 #q = 21 is the length of the alphabet.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to compute the weights for each sequence and the effective number of sequences.

	L = La + Lb
	M_train = size(training_set, 1) #M_train is the number of sequences in the training set.
	W, Meff = compWeights(dij_A, dij_B, training_set, M_train, L, sim_threshold)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to compute the reweighted frequencies.

	Pi_true, Pij_true = compute_freq(alignA_Num, alignB_Num, training_set, W, Meff, M_train, La, Lb, q)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to add the pseudocount to frequencies.

	Pi, Pij = pseudocount_freq(Pi_true, Pij_true, L, q, pseudocount_weight)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to compute PMIs from the matrices of frequencies.

	PMIs = get_PMIs(Pi, Pij, L, q)

	return PMIs, Meff

end


#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Computes PMIs from the matrices of frequencies.
"""

function get_PMIs(Pi::Array{Float64,2}, Pij::Array{Float64,4}, L::Int64, q::Int8)

	#Pij are the two-point frequencies.
	#Pi are single-point frequencies.
	#L is the number of aminoacids in the joint A-B alignment.
	#q = 21 is the length of the alphabet.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to PMIs from the matrices of frequencies. The last aa type(=gap) is ignored.

	PMIs = zeros(L, L, q, q)

	for i = 1:L, j = 1:L, alpha = 1:q-1, beta = 1:q-1
		PMIs[i,j,alpha,beta] = -log(Pij[i,j,alpha,beta]/(Pi[i,alpha] * Pi[j,beta]))
	end

	return PMIs

end
