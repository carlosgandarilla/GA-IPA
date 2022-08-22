"""
SA_replicates is a replicate of the SA updating algorithm for the GA problem.
"""

function SA_replicates(T_0::Float64, alpha::Float64, n_sweep::Int64, M::Int64, N::Int64, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1}, MSeqSpec::Array{Int64,1}, IndexSeqSpec::Array{Int64,1}, wij_A::SparseMatrixCSC{Float64,Int64}, wij_B::SparseMatrixCSC{Float64,Int64})

	#T_0 is the initial temperatures of the exponential schedule.
	#alpha is a factor between 0 and 1 to set the temperature T = T_0 * alpha^n_sweep.
	#n_sweep is the number of sweeps, a sweep is defined as N pairing updates.
	#M is the number of sequences in the dataset.
	#N is the number of species.
	#FirstSeqSpec is the label of the first sequence per species.
	#LastSeqSpec is the label of the last sequence per species.
	#MSeqSpec is the number of sequences per specie.
	#IndexSeqSpec is the index of the species for each sequence.
	#wij_A is the weigth for the edge graph of protein A datasets.
	#wij_B is the weigth for the edge graph of protein A datasets.

	dataout = zeros(M + 2)

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to generate a initail match randomly and...
	#...to compute the energy of the random match.

	protBmatch = randmatch(N, FirstSeqSpec, LastSeqSpec)

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to run Simulated Anneliang to align the graphs.

	SA_swept(protBmatch, T_0, alpha, n_sweep, M, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, wij_A, wij_B)

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to compute the TP true positive values.

	TP = truePos(M, protBmatch)
	e_pi = energ_graph(protBmatch, M, wij_A, wij_B)

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to save the TP values, the energy of the final match and the final match.

	dataout[1] = TP
	dataout[2] = e_pi
	dataout[3:end] = protBmatch

	return dataout

end




#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
SA does a given number of sweep, in each sweep update the temperature following...
...an exponential schedule and then Mseq MC updates.
"""

function SA_swept(protBmatch::Array{Int64,1}, T_0::Float64, alpha::Float64, n_sweep::Int64, M::Int64, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1}, MSeqSpec::Array{Int64,1}, IndexSeqSpec::Array{Int64,1}, wij_A::SparseMatrixCSC{Float64,Int64}, wij_B::SparseMatrixCSC{Float64,Int64})

	#protBmatch is the initial match between protein A and B.
	#T_0 is the initial temperatures of the exponential schedule.
	#alpha is a factor between 0 and 1 to set the temperature T = T_0 * alpha^n_sweep.
	#n_sweep is the number of sweeps, a sweep is defined as N pairing updates.
	#M is the number of sequences in the datasets.
	#FirstSeqSpec is the label of the first sequence per species.
	#LastSeqSpec is the label of the last sequence per species.
	#MSeqSpec is the number of sequences per specie.
	#IndexSeqSpec is the index of the species for each sequence.
	#wij_A is the weigth for the edge graph of protein A datasets.
	#wij_B is the weigth for the edge graph of protein A datasets.

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to loop over the number of sweeps.

	for i in 1:n_sweep

		T = T_0 * alpha^(i - 1)

		#-------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second step is to do swept i.e. loop over "Mseq" attemps of partner switch.

		for j in 1:M

			mcstep(protBmatch, T, M, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, wij_A, wij_B)

		end

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
"""
mcstep is a Monte Carlo step for the GA problem. Update the match if the new...
...match is energetically favorable (DeltaE_pi < 0) or with a probability...
...exp(- DeltaE_pi/T).
"""

function mcstep(protBmatch::Array{Int64,1}, T::Float64, M::Int64, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1}, MSeqSpec::Array{Int64,1}, IndexSeqSpec::Array{Int64,1}, wij_A::SparseMatrixCSC{Float64,Int64}, wij_B::SparseMatrixCSC{Float64,Int64})

	#protBmatch is the initial match between protein A and B.
	#T is the temperature at the MCMC step.
	#M is the number of sequences in the datasets.
	#FirstSeqSpec is the label of the first sequence per species.
	#LastSeqSpec is the label of the last sequence per species.
	#MSeqSpec is the number of sequences per specie.
	#IndexSeqSpec is the index of the species for each sequence.
	#wij_A is the weigth for the edge graph of protein A datasets.
	#wij_B is the weigth for the edge graph of protein A datasets.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to find the indices of sequences that will be permuted.

	bng, nd = seqperm(M, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to compute the Delta Energy for this permutation.

	DeltaE_pi = DeltaEnerg_graph(bng, nd, protBmatch, wij_A, wij_B)

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to switch partners inside a species.

	if DeltaE_pi < 0

		protB_bng = protBmatch[bng]
		protB_nd = protBmatch[nd]
		protBmatch[bng] = protB_nd
		protBmatch[nd] = protB_bng

	elseif exp(- DeltaE_pi/T) > rand()

		protB_bng = protBmatch[bng]
		protB_nd = protBmatch[nd]
		protBmatch[bng] = protB_nd
		protBmatch[nd] = protB_bng

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
seqperm generates two indices belonging to protein A that will be mismatched...
...Takes into account that indices belong to the same species.
"""

function seqperm(M::Int64, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1}, MSeqSpec::Array{Int64,1}, IndexSeqSpec::Array{Int64,1})

	#M is the number of sequences in the datasets.
	#FirstSeqSpec is the label of the first sequence per species.
	#LastSeqSpec is the label of the last sequence per species.
	#MSeqSpec is the number of sequences per specie.
	#IndexSeqSpec is the index of the species for each sequence.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third.1 step is to find the indices of sequences that will be mismatched...
	#...by randomly select a sequence from the dataset because it takes into account...
	#...that species with more sequences are more likely to be returned.

	bng = 0
	nd = 0
	count_perm = 0

	while count_perm < 1

		bng = rand(1:M)
		spec = IndexSeqSpec[bng] #spec is the species in which the permutation will take place.

		if MSeqSpec[spec] != 1 #Species with one sequences are avoided.
			nd = rand(setdiff(FirstSeqSpec[spec]:LastSeqSpec[spec], bng))
			count_perm += 1
		end

	end

	return bng, nd

end




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
truePos returns the fraction of correct matches.
"""

function truePos(M::Int64, protB::Array{Int64,1})

	#Mseq is the number of sequences in the datasets.
	#protB is a given match.

	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is to compute the TP true positive values.

	countTP = 0

	for i in 1:M

		countTP += protB[i] == i

	end

	return countTP/M

end




#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
DeltaEnerg_graph returns the Delta energy that comes when two matched sequences are permuted.
"""

function DeltaEnerg_graph(bng::Int64, nd::Int64, Match_protB::Array{Int64,1}, wij_A::SparseMatrixCSC{Float64,Int64}, wij_B::SparseMatrixCSC{Float64,Int64})

	#bng is the label of first sequence involved in the switch.
	#nd is the label of second sequence involved in the switch.
	#Match_protB are the labels of protein B that match with protein A (that always goes from 1:5107) i.e. represents a match.
	#wij_A is the edge weight in the kNN graph for protein A.
	#wij_B is the edge weight in the kNN graph for protein B.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is define "M_seq" the number of nodes in the kNN graph and other important things.

	e_pi = 0.0 #e_pi is the energy.

	rows_A = rowvals(wij_A) #rows_A is a vector of the row indices of "wij_A". If there are "n" values stored in first column of "wij_A", then...
	                        #...the first "n" values of "rows_A" are the rows where this values are stored and so on. So, the "M_seq" columns represent...
							#...the nodes and "rows_A" contains all the links of these nodes. To take the links for a given node "i", we have to use...
							#..."nzrange(wij_A, i)".
	vals_A = nonzeros(wij_A) #vals_A is a vector of the structural nonzero values in sparse array "wij_A". It works idem to "rows_A" but store the...
	                         #...values of "wij_A" and not the row positions.
	rows_B = rowvals(wij_B)
	vals_B = nonzeros(wij_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The second step is to find the partners of proteins "bng" and "nd".

	pi_bng = Match_protB[bng]
	pi_nd = Match_protB[nd]

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The third step is to loop over the edges of sequence "bng".

	for m in nzrange(wij_A, bng)

		j_bng = rows_A[m]

		pi_j = Match_protB[j_bng] #pi_j is the protein B label that match sequence "j_bng" of protein A, this result is given by the matching function "pi_j = pi(j_bng)".

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.1 step is to loop over the edges of sequence "pi_bng".

		for n in nzrange(wij_B, pi_bng)

			edgeB_bng = rows_B[n]

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.1.1 step is to put a condition to check that the partner pi_j = pi(j_bng) of the edge "j_bng" of node "bng" is and edge of the partner pi_bng = pi(bng).

			if pi_j == edgeB_bng && edgeB_bng != pi_nd
				e_pi += vals_A[m] * vals_B[n]
			end

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The third.2 step is to loop over the edges of sequence "pi_nd".

		for n in nzrange(wij_B, pi_nd)

			edgeB_nd = rows_B[n]

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.2.1 step is to put a condition to check that the partner pi_j = pi(j_bng) of the edge "j_bng" of node "bng" is and edge of the partner pi_bng = pi(bng).

			if pi_j == edgeB_nd && edgeB_nd != pi_bng
				e_pi += - vals_A[m] * vals_B[n]
			end

		end

	end

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The fourth step is to loop over the edges of sequence "nd".

	for m in nzrange(wij_A, nd)

		j_nd = rows_A[m]

		pi_j = Match_protB[j_nd] #pi_j is the protein B label that match sequence "j_nd" of protein A, this result is given by the matching function "pi_j = pi(j_nd)".

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The fourth.1 step is to loop over the edges of sequence "pi_bng".

		for n in nzrange(wij_B, pi_bng)

			edgeB_bng = rows_B[n]

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The fourth.1.1 step is to put a condition to check that the partner pi_j = pi(j_nd) of the edge "j_nd" of node "nd" is and edge of the partner pi_bng = pi(bng).

			if pi_j == edgeB_bng && edgeB_bng != pi_nd
				e_pi += - vals_A[m] * vals_B[n]
			end

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The fourth.2 step is to loop over the edges of sequence "pi_nd".

		for n in nzrange(wij_B, pi_nd)

			edgeB_nd = rows_B[n]

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			#The third.2.1 step is to put a condition to check that the partner pi_j = pi(j_nd) of the edge "j_nd" of node "nd" is and edge of the partner pi_bng = pi(bng).

			if pi_j == edgeB_nd && edgeB_nd != pi_bng
				e_pi += vals_A[m] * vals_B[n]
			end

		end

	end

	return e_pi

end




#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
energ_graph returns the energy for a given (whole) match between proteins A and B.
"""

function energ_graph(Match_protB::Array{Int64,1}, M::Int64,  wij_A::SparseMatrixCSC{Float64,Int64}, wij_B::SparseMatrixCSC{Float64,Int64})

	#Match_protB are the labels of protein B that match with protein A (that always goes from 1:5107) i.e. represents a match.
	#M is the number of sequences in the datasets.
	#wij_A is the edge weight in the kNN graph for protein A.
	#wij_B is the edge weight in the kNN graph for protein B.

	e_pi = 0.0 #e_pi is the energy.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The first step is define "M_seq" the number of nodes in the kNN graph and other important things.

	rows_A = rowvals(wij_A) #rows_A is a vector of the row indices of "wij_A". If there are "n" values stored in first column of "wij_A", then...
	                        #...the first "n" values of "rows_A" are the rows where this values are stored and so on. So, the "M_seq" columns represent...
							#...the nodes and "rows_A" contains all the links of these nodes. To take the links for a given node "i", we have to use...
							#..."nzrange(wij_A, i)".
	vals_A = nonzeros(wij_A) #vals_A is a vector of the structural nonzero values in sparse array "wij_A". It works idem to "rows_A" but store the...
	                         #...values of "wij_A" and not the row positions.
	rows_B = rowvals(wij_B)
	vals_B = nonzeros(wij_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The second step is to loop over the "M_seq" nodes of protein A (columns of wij_A).

	for i in 1:M

	    pi_i = Match_protB[i] #pi_i is the protein B label that match sequence "i" of protein A, this result is given by the matching function "pi_i = pi(i)".

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	    #The third step is to loop over the edges of sequence "i".

		for k in nzrange(wij_A, i)

			j = rows_A[k]

			pi_j = Match_protB[j] #pi_j is the protein B label that match sequence "edge_i" of protein A, this result is given by the matching function "pi_j = pi(edge_i)".

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		    #The fourth step is to loop over the edges of sequence "pi_i".

			for l in nzrange(wij_B, pi_i)

				edge_l = rows_B[l]

				#---------------------------------------------------------------------------------------------------------------------------------------------------------------
			    #The fifth step is to put a condition to check that the partner pi_j = pi(j) of the edge "edge_i" of node "i" is and edge of the partner pi_i = pi(i).

				if pi_j == edge_l

					e_pi += vals_A[k] * vals_B[l]

				end

			end

		end

	end

	return -e_pi/2.0

end




#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
randmatch returns a whole match, it randomly selects partners inside species.
...A match is vector with the indices of protein B sequences, that match with...
...protein A indices that goes from 1:Mseq, but the indices in protein B takes...
...into account the species (matching between sequeces of different species are forbidden).
"""

function randmatch(N::Int64, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1})

	#N is the number of sequences per species.
	#FirstSeqSpec is the index of the first sequence for each species
	#LastSeqSpec is the number/label of the last sequence per species.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to randomly match all sequences per species.

	prot_B = Int64[]

	for i in 1:N

		prot_B = union(prot_B, shuffle(FirstSeqSpec[i]:LastSeqSpec[i]))

	end

	return prot_B

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
