"""
Orthology_Prop returns two sparse (M x M) matrices for both proteins A and B. The entries are the weight of links
between nodes i and j, in this case the links are given by orthologs.
"""

function Orthology_Prop(N::Int64, M::Int64, MSeqSpec::Array{Int64,1}, IndexSeqSpec::Array{Int64,1}, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1}, dij_A::Array{Int64,2}, dij_B::Array{Int64,2})

	#M is the number of species.
	#M is the number of sequences in the datasets.
	#MSeqSpec is the number of sequences per species.
	#IndexSeqSpec is the index of the species for each sequence.
	#LastSeqSpec is the number/label of the last sequence per species.
	#dij_A is the Hamming distance between sequences in the protein A datasets.
	#dij_B is the Hamming distance between sequences in the protein B datasets.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to find for each sequence the most similar (in terms of Hamming distance) sequence in each species.
	#...So, each node have M_spec links (similar nodes), one for each species.

	closestEdges_A, closestEdges_B = closestEdges_Spec(N, M, FirstSeqSpec, LastSeqSpec, dij_A, dij_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The third step is to find orthologs by finding how many times the closest sequence in each species is reciprocal,
	#... i.e. find if the closest partner j of the sequence i has also i as his closest partner.

	orthologs_A, orthologs_B = Orthologs(N, M, IndexSeqSpec, closestEdges_A, closestEdges_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The fourth step is to compute the normalization distance dk for proteins A and B.

	dk_A, dk_B = normDist_Orth(M, orthologs_A, orthologs_B, dij_A, dij_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The fifth step is to compute the links/edges weight for each node of the orthology graph.

	weight_A, weight_B = weight_Orth(M, orthologs_A, orthologs_B, dk_A, dk_B, dij_A, dij_B)

	return weight_A, weight_B

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
weight_Orth returns two (M_seq x M_seq) sparse matrices for proteins A and B, whose only...
...non zero elements are the weight of the links. So, if there are an element weight_A[i,j]...
...diferent of zero it represent that there is a link between node "i" and "j" and its value...
...the weight.
"""


function weight_Orth(M::Int64, orthologs_A::SparseMatrixCSC{Int64,Int64}, orthologs_B::SparseMatrixCSC{Int64,Int64}, dk_A::Float64, dk_B::Float64, dij_A::Array{Int64,2}, dij_B::Array{Int64,2})

	#M is the number of sequences.
	#orthologs_A, orthologs_B are the orthologs of both comunities A and B.
	#dk_A, dk_B are the normalization distances of both communities A and B.
	#dij_A, dij_B are the Hamming distance matrices of both comunities A and B.

	rows_A = rowvals(orthologs_A)
	vals_A = nonzeros(orthologs_A)

	rows_B = rowvals(orthologs_B)
	vals_B = nonzeros(orthologs_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to compute the links/edges weight for each node of the orthology graph.

	weight_A = spzeros(Float64, M, M)
	weight_B = spzeros(Float64, M, M)

	@inbounds for j in 1:M

		for m in nzrange(orthologs_A, j)
			weight_A[vals_A[m], j] = exp(-(dij_A[vals_A[m], j]/dk_A)^2)
		end

		for n in nzrange(orthologs_B, j)
			weight_B[vals_B[n], j] = exp(-(dij_B[vals_B[n], j]/dk_B)^2)
		end

	end

	return weight_A, weight_B

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
normDist_Orth returns the normalization distance d_k for the orthology graph of both communities A and B.
"""


function normDist_Orth(M::Int64, orthologs_A::SparseMatrixCSC{Int64,Int64}, orthologs_B::SparseMatrixCSC{Int64,Int64}, dij_A::Array{Int64,2}, dij_B::Array{Int64,2})

	#M is the number of sequences.
	#orthologs_A, orthologs_B are the orthologs of both comunities A and B.
	#dij_A, dij_B are the Hamming distance matrices of both comunities A and B.

	rows_A = rowvals(orthologs_A)
	vals_A = nonzeros(orthologs_A)

	rows_B = rowvals(orthologs_B)
	vals_B = nonzeros(orthologs_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to compute the normalization distance dk for both comunities A and B.

	dk_A = 0
	dk_B = 0

	@inbounds for j in 1:M

		#-----------------------------------------------------------------------------------------------------------------------------------------------------------
	    #The first.1 step is to find the most distant neighbor for each sequence "j" between its orthologs.

		dk_a = 0
		dk_b = 0

		for m in nzrange(orthologs_A, j)
			dij = dij_A[vals_A[m], j]
			if dij > dk_a
				dk_a = dij
			end
		end

		for n in nzrange(orthologs_B, j)
			dij = dij_B[vals_B[n], j]
			if dij > dk_b
				dk_b = dij
			end
		end

		dk_A += dk_a
		dk_B += dk_b

	end

	return dk_A/M, dk_B/M

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Orthologs returns two (N x M) matrices for proteins A and B. Each column of the matrices contains the indeces of
orthologs for a given node. The number of nodes/sequences are M_seq and the maximum number of orthologs are N - 1 the
number of species. The zero entries of the matrix represents that there is not othologs between the given node(column) and
the species(row). An orthologs can be found by finding how many times the closest sequence in each species is reciprocal, i.e
by finding if the closest partner j of the sequence i has also i as his closest partner.
"""


function Orthologs(N::Int64, M::Int64, IndexSeqSpec::Array{Int64,1}, closestEdges_A::Array{Int64,2}, closestEdges_B::Array{Int64,2})

	#N is the number of species.
	#M is the number of sequences in the datasets.
	#IndexSeqSpec is the index of the species for each sequence.
	#closestEdges_A contains the most similar sequence in each species for each sequence of protein A.
	#closestEdges_B contains the most similar sequence in each species for each sequence of protein B.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------

	orthologs_A = spzeros(Int64, N, M) #Orthologs_A contains the orthologs of each node in community A.
	orthologs_B = spzeros(Int64, N, M) #Orthologs_B contains the orthologs of each node in community B.

	@inbounds for j in 1:M, i in IndexSeqSpec[j] + 1:N

		if closestEdges_A[IndexSeqSpec[j], closestEdges_A[i, j]] == j
			orthologs_A[i, j] = closestEdges_A[i, j]
			orthologs_A[IndexSeqSpec[j], closestEdges_A[i, j]] = j
		end

		if closestEdges_B[IndexSeqSpec[j], closestEdges_B[i, j]] == j
			orthologs_B[i, j] = closestEdges_B[i, j]
			orthologs_B[IndexSeqSpec[j], closestEdges_B[i, j]] = j
		end

	end

	return orthologs_A, orthologs_B

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
closestEdges_Spec returns two (N x M) matrices for both proteins A and B. Each matrix contains, for each node/sequence,
the most similar (in terms of Hamming distance) sequence in each species. So, each column has N links (similar nodes)
for each M node, one link for each species.
"""


function closestEdges_Spec(N::Int64, M::Int64, FirstSeqSpec::Array{Int64,1}, LastSeqSpec::Array{Int64,1}, dij_A::Array{Int64,2}, dij_B::Array{Int64,2})

	#N is the number of species.
	#M is the number of sequences in the datasets.
	#FirstSeqSpec is the number/label of the first sequence per species.
	#LastSeqSpec is the number/label of the last sequence per species.
	#dij_A is the Hamming distance between sequences in the protein A datasets.
	#dij_B is the Hamming distance between sequences in the protein B datasets.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to find for each sequence the most similar (in terms of Hamming distance) sequence in each species.
	#...So, each node have M_spec links (similar nodes), one for each species.

	closestEdges_A = Array{Int64}(undef, N, M)
    closestEdges_B = Array{Int64}(undef, N, M)

	@inbounds for j in 1:M, i in 1:N

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The second step is to find the minimun value of the Hamming distance matrix dij in the row i, and the columns corresponding to especies j and...
		#...to copy the index of this minimun value of dij in the row i of closestEdges_HK as a direct link of i.

		closestEdges_A[i, j] = FirstSeqSpec[i] + argmin(view(dij_A, FirstSeqSpec[i]:LastSeqSpec[i], j)) - 1
		closestEdges_B[i, j] = FirstSeqSpec[i] + argmin(view(dij_B, FirstSeqSpec[i]:LastSeqSpec[i], j)) - 1

	end

	return closestEdges_A, closestEdges_B

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
