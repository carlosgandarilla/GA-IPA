#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
predict_pairs_hungarian makes pairing predictions using a global algorithm like the Hungarian algorithm, and rank assignment following also a global score.
"""

function predict_pairs_hungarian_MI(alignA_Num::Array{Int8,2}, alignB_Num::Array{Int8,2}, testing_set::Array{Int64,2}, M_test::Int64, PMIs::Array{Float64,4}, La::Int64, Lb::Int64)

	#alignA_Num contains alignment A after the aminoacids being converted into numbers, on which the model is trained.
	#alignB_Num contains alignment B after the aminoacids being converted into numbers, on which the model is trained.
	#testing_set is an array containing the id of sequences As and Bs from an species that will be matched.
	#M_test is the number of sequences in a "testing_set" .
    #PMIs is the mutual information for joint A-B alignment.
	#La is the number of aminoacids in the alignment A.
	#Lb is the number of aminoacids in the alignment B.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to initialize the Results array, used for saving data
	#col 1: species
	#col 2: A index in initial alignment
	#col 3: B index in initial alignment
	#col 4: score by absolute energy of pairing
	#col 5: gap

	Results = zeros(M_test, 5)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to loop over species in the testing set

	totcount = 0 #total pair counter

	for i in unique(testing_set[:, 1]) #1:size(table_count_species_test_set,1)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.1 step is to extract those sequence belonging to the same species.

		test_seqs, count_test_seqs = GetSeqFromTestSet(testing_set, M_test, i)

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.2 step is to now compute the interaction energies of all the A-B pairs within the species corresponding to i

		pairing_scores = compute_pairing_scores_MI(alignA_Num, alignB_Num, test_seqs, count_test_seqs, PMIs, La, Lb)

    	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.3 step is to ...

		if count_test_seqs == 1
        	assignment = 1
        	pairing_scores_b = pairing_scores .- minimum(pairing_scores) #ensure that all elements are >=0
    	elseif minimum(pairing_scores) == maximum(pairing_scores)
        	assignment = sample(1:count_test_seqs, count_test_seqs; replace=false) #avoids spurious positive results
            pairing_scores_b = pairing_scores .- minimum(pairing_scores) #ensure that all elements are >=0
    	else #use the Hungarian algorithm
        	pairing_scores_b = pairing_scores .- minimum(pairing_scores) #ensure that all elements are >=0
        	assignment, score = hungarian(pairing_scores_b)
    	end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.4 step is to compute the predicted assignment using Hungarian algorithm.

		bigval = 1000 * abs(maximum(pairing_scores_b))

	 	for j = 1:count_test_seqs

        	totcount += 1
        	Results[totcount, 1] = i
        	Results[totcount, 2] = test_seqs[j, 1] #initial index of A sequence (A: line)
        	Results[totcount, 3] = test_seqs[assignment[j], 2] #initial index of B sequence (B: line)
        	Results[totcount, 4] = pairing_scores[j, assignment[j]] #absolute energy of the pairing
			if count_test_seqs == 1
            	Results[totcount, 5] = abs(pairing_scores) #no real gap... consider that absolute energy is gap
        	elseif minimum(pairing_scores) == maximum(pairing_scores)
            	#no gap for this assignment
            	Results[totcount, 5] = 0
        	else
            	#calculate gap for this assignment
            	pairing_scores_mod = copy(pairing_scores_b)
            	pairing_scores_mod[j, assignment[j]] = bigval
            	score_mod = hungarian(pairing_scores_mod)[2]
            	Results[totcount, 5] = score_mod - score
        	end

		end

	end

	return Results

end


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
predict_pairs_gap makes pairing predictions using a greedy algorithm that minimize the energy of the pairs and rank assigment by energy gap.
"""

function predict_pairs_gap(alignA_Num::Array{Int8,2}, alignB_Num::Array{Int8,2}, testing_set::Array{Int64,2}, M_test::Int64, W_store::Array{Float64,4}, La::Int64, Lb::Int64)

	#alignA_Num contains alignment A after the aminoacids being converted into numbers, on which the model is trained.
	#alignB_Num contains alignment B after the aminoacids being converted into numbers, on which the model is trained.
	#testing_set is an array containing the id of sequences in an species, i.e. id of seq A and id of seq B.
	#M_test is the number of sequences in a "test_seqs" (subset of the testing set). .
    #W_store are the MF couplings of the joint A-B alignment.
	#La is the number of aminoacids in the alignment A.
	#Lb is the number of aminoacids in the alignment B.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to initialize the Results array, used for saving data
	#col 1: species
	#col 2: A index in initial alignment
	#col 3: B index in initial alignment
	#col 4: score by absolute energy of pairing
	#col 5: gap wrt A
	#col 6: gap wrt B

	Results = zeros(M_test, 6)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to loop over species in the testing set

	totcount = 0 #total pair counter

	for i in unique(testing_set[:, 1])

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.1 step is to extract those sequence belonging to the same species.

		test_seqs, count_test_seqs = GetSeqFromTestSet(testing_set, M_test, i)
		DeletedInd = fill(0, count_test_seqs, 2)
		count = 0

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.2 step is to compute the interaction energies of all the A-B pairs within the species corresponding to i

		HKRR_energy = compute_energies(alignA_Num, alignB_Num, test_seqs, count_test_seqs, W_store, La, Lb)
		HKRR_energy_ini = copy(HKRR_energy)

    	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second.3 step is to make pairings based on energy within this species (first pair the HK&RR that have the lowest energy etc.)
		#The trivial case never occurs (when there is only one pair in this species).
		#There is always more than one pair in this species.

		bigval = 1000 * abs(maximum(HKRR_energy))
		val = minimum(HKRR_energy)
		#max_energy = maximum(HKRR_energy)
		#HKRR_energy_b = HKRR_energy .- min_energy #ensure that all elements are >=

		while val < bigval #not all the HK and RR have been eliminated: there are still pairings to be made, pursue

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#indices where the min energy is found => make this pair
			inds_min = findall(HKRR_energy .== val)

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#compute energy gaps associated to these possible pairings
			theGaps = zeros(size(inds_min, 1), 5)
			#col 1: A index in initial alignment
			#col 2: B index in initial alignment
			#col 3: gap wrt A
			#col 4: gap wrt B
			#col 5: minimum gap

			for myind = 1:size(inds_min, 1)

				#---------------------------------------------------------------------------------------------------------------------------------------------------------------
				#cartesian indexes of minimum value in case there are more than one (size(inds_min, 1) > 1)
				theGaps[myind, 1] = inds_min[myind][1]
            	theGaps[myind, 2] = inds_min[myind][2]

				#---------------------------------------------------------------------------------------------------------------------------------------------------------------
				#first compute gap wrt HK, i.e. within row
                theLine = HKRR_energy[inds_min[myind][1], :]
                theLine[inds_min[myind][2]] = bigval
                val2 = minimum(theLine) #next smallest energy of a coupling for this HK (behind val)
                if val2 == bigval #there is only one RR left that can pair with this HK. Gap notion is ill-defined.
                    theGaps[myind, 3] = -1
                else
                    theGaps[myind, 3] = val2 - val
                    better = HKRR_energy_ini[inds_min[myind][1], DeletedInd[1:count, 2]]
                    better = findall(better .< val)
                    if length(better) != 0 #the best match(es) have been deleted by previous matches - penalize
                        theGaps[myind, 3] = (val2 - val) ./ (length(better) + 1)
                    end
                end

				#---------------------------------------------------------------------------------------------------------------------------------------------------------------
				#then compute gap wrt RR, i.e. within col
                theCol = HKRR_energy[:, inds_min[myind][2]]
                theCol[inds_min[myind][1]] = bigval
                val2 = minimum(theCol) #next smallest energy of a coupling for this RR (behind val)
                if val2 == bigval #there is only one HK left that can pair with this RR. Gap notion is ill-defined.
                    theGaps[myind, 4] = -1
                else
                    theGaps[myind, 4] = val2 - val
                    better = HKRR_energy_ini[DeletedInd[1:count, 1], inds_min[myind][2]]
                    better = findall(better .< val)
                    if length(better) != 0 #the best match(es) have been deleted by previous matches - penalize
                         theGaps[myind, 4] = (val2 - val) ./ (length(better) + 1)
                    end
                end

				#---------------------------------------------------------------------------------------------------------------------------------------------------------------
				#also store min gap
                theGaps[myind, 5] = min(theGaps[myind, 3], theGaps[myind, 4])

			end

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------

			if size(inds_min, 1) > 1
                #rank the ex-aequo best pairs by min gap
                auxGap = findall(theGaps[:, 5] .== minimum(theGaps[:, 5]))
                if size(auxGap, 1) == 1
                    theGaps = theGaps[auxGap, :]
                else #pick a random pair
                    theGaps = theGaps[rand(auxGap), :]
                end
            end

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#make one pair and then move on
            totcount += 1
            count += 1

            Results[totcount, 1] = i
            Results[totcount, 2] = test_seqs[convert(Int64, theGaps[1]), 1] #initial index of HK sequence (HK: line)
            Results[totcount, 3] = test_seqs[convert(Int64, theGaps[2]), 2] #initial index of RR sequence (RR: col)
            Results[totcount, 4] = val #absolute energy of the pairing - was the min of the HKRR_energy matrix
            Results[totcount, 5] = theGaps[3] #gap within row - wrt this HK
            Results[totcount, 6] = theGaps[4] #gap within col - wrt this RR

            DeletedInd[count, 1] = convert(Int64, theGaps[1])
            DeletedInd[count, 2] = convert(Int64, theGaps[2])

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
 			#since we only want one-to-one pairings, the HK and RR can't interact with anything: suppress them from further consideration
            HKRR_energy[convert(Int64, theGaps[1]), :] .= bigval #replace the row by bigval to effectively suppress HK from further consideration
            HKRR_energy[:, convert(Int64, theGaps[2])] .= bigval #replace the col by bigval to effectively suppress RR from further consideration

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
 			#now update val for next round!
            val = minimum(HKRR_energy)

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#Almost done with this species - all pairings have been made
        #Just fix the undefined gaps

		aux_5 = Results[totcount - count + 1:totcount, 5]
		aux_6 = Results[totcount - count + 1:totcount, 6]
		save_aux5_ind = 0
		save_aux6_ind = 0
		for j = 1:count
			if aux_5[j] == -1
				save_aux5_ind = j
				aux_5[j] = Inf
			end
			if aux_6[j] == -1
				save_aux6_ind = j
				aux_6[j] = Inf
			end
		end
		#println("totcount = ", totcount, " count = ", count, " save_aux5_ind = ", save_aux5_ind)
		#println("totcount = ", totcount, " count = ", count, " save_aux6_ind = ", save_aux6_ind)
		if save_aux5_ind != 0 && minimum(aux_5) != Inf
			Results[totcount - count + save_aux5_ind, 5] = minimum(aux_5)
		elseif save_aux5_ind != 0 && minimum(aux_5) == Inf
			Results[totcount - count + save_aux5_ind, 5] = 0
		end
		if save_aux6_ind != 0 && minimum(aux_6) != Inf
			Results[totcount - count + save_aux6_ind, 6] = minimum(aux_6)
		elseif save_aux6_ind != 0 && minimum(aux_6) == Inf
			Results[totcount - count + save_aux6_ind, 6] = 0
		end

	end

	return Results

end


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
GetSeqFromTestSet gets the sequences from the testing set that belongs to the same species.
"""

function GetSeqFromTestSet(testing_set::Array{Int64, 2}, M_test::Int64, i::Int64)

	test_seqs_A = Int64[]
	test_seqs_B = Int64[]
	count_test_seqs = 0

	for l = 1:M_test
		if testing_set[l, 1] == i
			count_test_seqs += 1
			push!(test_seqs_A, testing_set[l, 2])
			push!(test_seqs_B, testing_set[l, 3])
		end
	end

	return hcat(test_seqs_A, test_seqs_B), count_test_seqs

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
