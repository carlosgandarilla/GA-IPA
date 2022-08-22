"""
kNN_Prop returns two sparse (M x M) matrices for both proteins A and B. The entries are the weight of links
between nodes i and j, in this case the links are given by k-NN (nearest neighbor).
"""

function kNNprop(kNN::Int64, M::Int64, dij_A::Array{Int64,2}, dij_B::Array{Int64,2})

	#kNN is the minimum number of neighbors per node.
	#M_seq is the number of sequences in the datasets.
	#dij_A is the Hamming distance between sequences in the protein A datasets.
	#dij_B is the Hamming distance between sequences in the protein B datasets.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to find the direct edges/links for each node of kNN graph.

	directEdges_A, directEdges_B, dk_A, dk_B = directEdges(kNN, M, dij_A, dij_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The second step is to compute the links/edges weight for each node of k-NN graph.

	weight_A = spzeros(Float64, M, M)
	weight_B = spzeros(Float64, M, M)

	@inbounds for j in 1:M, i in 1:kNN

		weight_A[directEdges_A[i, j], j] = exp(-(dij_A[directEdges_A[i, j], j]/dk_A)^2)
		weight_A[j, directEdges_A[i, j]] = weight_A[directEdges_A[i, j], j]

		weight_B[directEdges_B[i, j], j] = exp(-(dij_B[directEdges_B[i, j], j]/dk_B)^2)
		weight_B[j, directEdges_B[i, j]] = weight_B[directEdges_B[i, j], j]

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
directEdges returns the direct edges/links for each node of kNN graph. There is an edge/link between two nodes if they are similar in...
...terms of Hamming distance. The direct link from node i to node j is given when node j is similar to node i but, this don't has to be....
...reciprocal: the node i is similar to node j. So, a node i has at least kNN direct edges, but there can be another nodes linked with i.
"""

function directEdges(kNN::Int64, M::Int64, dij_A::Array{Int64,2}, dij_B::Array{Int64,2})

	#kNN is the minimum number of neighbors per node.
	#M_seq is the number of sequences in the datasets.
	#dij_A is the Hamming distance between sequences in the protein A datasets.
	#dij_B is the Hamming distance between sequences in the protein B datasets.

	dijA = copy(dij_A)
	dijB = copy(dij_B)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to find the direct edges/links for each node of 21NN graph. There is an edge/link between two nodes if they are similar in...
	#...terms of Hamming distance. The direct link from node i to node j is given when node j is similar to node i but, this don't has to be....
	#...reciprocal: the node i is similar to node j. So, a node k has at least 21 direct edges, but there can be another nodes linked with k.

	dk_A = 0
	dk_B = 0

	directEdges_A = Array{Int64}(undef, kNN, M) #directEdges_HK contains the 21 direct links of each HK node.
	directEdges_B = Array{Int64}(undef, kNN, M) #directEdges_RR contains the 21 direct links of each RR node.

	@inbounds for j in 1:M

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	    #The first.1 step is to avoid the selection of the diagonal element.

		dijA[j, j] = 10^8
		dijB[j, j] = 10^8

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	    #The first.2 step is to loop over the number of nearest neighbors except the last one (kNN).

		for i in 1:kNN - 1

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	        #The first.2.1 step is to find the minimun value of the Hamming distance matrix dij in the row i, and the index (column).

            directEdges_A[i, j] = argmin(view(dijA, :, j))
		    directEdges_B[i, j] = argmin(view(dijB, :, j))

			#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		    #The first.2.2 step is to replace this minimun value of dij for -Inf (a very low value) and this way we avoid the selection...
			#...of this value in the next iteration.

			dijA[directEdges_A[i, j], j] = 10^8
			dijB[directEdges_B[i, j], j] = 10^8

		end

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	    #The first.3 step is to repeat the process for the last nearest neighbors (kNN). The first step is to find the minimun value of...
		#...the Hamming distance matrix dij in the row i, and the index (column).

		closA_dij, ind_A = findmin(view(dijA, :, j))
		closB_dij, ind_B = findmin(view(dijB, :, j))

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
		#The first.3.2 step is to copy the index of this minimun value of dij in the row i of directEdges as a direct link of i.

		directEdges_A[kNN, j] = ind_A
		directEdges_B[kNN, j] = ind_B

		#---------------------------------------------------------------------------------------------------------------------------------------------------------------
	    #The first.3.3 step is to compute the normalization distance for the kNN graph, defined as the average distance between each protein and its k-th neighbor.

		dk_A += closA_dij
		dk_B += closB_dij

	end

	return directEdges_A, directEdges_B, dk_A/M, dk_B/M

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
