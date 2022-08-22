"""
datareader reads the joint alignment of protein pair A-B, and extracts vital information from both proteins A and B.
The entries of this functions are the name of the files where the data is stored and the number of amino acids for
both proteins A and B. Like for HK-RR dataset:

protfile = "./Concat_nnn_extr_withFirst.fasta"
La = 64 #La is the number of aminoacids in the HK protein.
Lb = 112 #Lb is the number of aminoacids in the RR protein.


datareader returns:

* "alignA_Num"
* "alignB_Num"
* "dij_A" the Hamming distance matrix for protein A,
* "dij_B" the Hamming distance matrix for protein B,
* "FirstSeqSpec" is the index of the first sequence for each species,
* "LastSeqSpec" is the index of the last sequence for each species,
* "MSeqSpec" is the number of sequences per species,
* "IndexSeqSpec" is the index of the species for each sequence,
* "M" is the number of sequences in the dataset,
* "N" is the number of species in the dataset.
"""

function datareader(protfile::String, La::Int64, Lb::Int64)

	#protfile is the file of joint alignment A-B proteins.
	#La is the number of aminoacids in the A protein.
	#Lb is the number of aminoacids in the B protein.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to import the dataset. For AF datasets, first and last sequences are dummies.

    jointAB = readfasta(protfile)
	M = length(jointAB) - 2 #M is the number of sequences in the concatenated HK-RR protein. irst and last are dummies.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The second step is to find the name of each sequences. Species are indicated explicitly in the headers, the letters between the first two "|".

    nAme = []

    for i = 2:M + 1

        bgn = findfirst("|", jointAB[i][1])[1] + 1
        nd = findnext("|", jointAB[i][1], bgn + 1)[1] - 1
        push!(nAme, jointAB[i][1][bgn:nd])

    end

	Spec_nAme = unique(nAme) #Spec_nAme are the name of the species.
	N = length(Spec_nAme) #N is the number of species.

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The third step is to ....

    data_A = Array{Any,2}(undef, M, 3)
	data_B = Array{Any,2}(undef, M, 3)
	data_concat = Array{Any,2}(undef, M, 3)
	MSeqSpec = Int64[] #MSeq_spec is a vector with the number of sequences per species.
	LastSeqSpec = Int64[] #MSeq_spec is a vector with the id of the last sequence in each species.

	cumul_count = 0 #count the number of cumulative sequences at the end of species (id of the last sequence in each species).
	count_Spec = 1 #count the especies in the loop.

    for i in Spec_nAme #loop over species
	    count = 0 #count sequences in each species.
		for j in 1:M #loop over all sequences in the dataset.
            if nAme[j] == i #check if sequence "j" belongs to species "i".
				count += 1
				#fill data of protein A
				data_A[cumul_count + count, 1] = jointAB[j + 1][2][1:La]
				data_A[cumul_count + count, 2] = count_Spec
				data_A[cumul_count + count, 3] = i
				#fill data of protein B
				data_B[cumul_count + count, 1] = jointAB[j + 1][2][La + 1:La + Lb]
				data_B[cumul_count + count, 2] = count_Spec
				data_B[cumul_count + count, 3] = i
				#fill data of joint protein A-B
				data_concat[cumul_count + count, 1] = jointAB[j + 1][2]
				data_concat[cumul_count + count, 2] = count_Spec
				data_concat[cumul_count + count, 3] = i
			end
		end
		cumul_count += count
		push!(MSeqSpec, count)
		push!(LastSeqSpec, cumul_count)
		count_Spec += 1
	end

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The fourth step is to find the number/label of the first sequence per species.

	FirstSeqSpec = LastSeqSpec - MSeqSpec + fill(1, N)

	#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The fifth step is to find IndexSeqSpec

	if data_A[:,2] == data_B[:,2]
		IndexSeqSpec = convert(Array{Int64, 1}, data_A[:,2])
	end

	#println(IndexSeqSpec)

	#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	#The sixth step is to compute pairwise Hamming distances.

	alignA_Num = numberedAlignment(data_A, M, 1, La)
	alignB_Num = numberedAlignment(data_B, M, 1, Lb)
	dij_A = Distances.pairwise(Hamming(), alignA_Num, dims = 1)
	dij_B = Distances.pairwise(Hamming(), alignB_Num, dims = 1)

	return alignA_Num, alignB_Num, dij_A, dij_B, FirstSeqSpec, LastSeqSpec, MSeqSpec, IndexSeqSpec, N, M

end



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
numbered_alignment reads the aminoacids letters of the alignment from inputfile, removes inserts and converts into numbers...
...The part of the alignment is defined by N_bng and N_end.
"""

function numberedAlignment(inputfile::Matrix{Any}, M::Int64, L_bng::Int64, L_end::Int64)

    #inputfile is the data file that contains anminoacid sequences and species' indexes and names.
    #M is the number of sequences in the concatenated the protein.
    #L_bng is the number of the aminoacids at which the convertion will start.
    #L_end is the number of the aminoacids at which the convertion will end.

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #The first step is to read the input file and to convert an anminoacid letters (an string) into a number from 1 to 21.

    align_Num = Array{Int8}(undef, M, L_end - L_bng + 1)

    for i in 1:M, j in L_bng:L_end

        align_Num[i, j] = letter2number(inputfile[i, 1][j])

    end

    return align_Num

end


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
letter2number is a function that converts an anminoacids letters (an string) into a number from 1 to 21.
"""

function letter2number(a::Char)

    #a is the aminoacids letters, an string.

    if a == "-"
        x = 21
    elseif a == 'A'
        x = 1
    elseif a == 'C'
        x = 2
    elseif a == 'D'
        x = 3
    elseif a == 'E'
        x = 4
    elseif a == 'F'
        x = 5
    elseif a == 'G'
        x = 6
    elseif a == 'H'
        x = 7
    elseif a == 'I'
        x = 8
    elseif a == 'K'
        x = 9
    elseif a == 'L'
        x = 10
    elseif a == 'M'
        x = 11
    elseif a == 'N'
        x = 12
    elseif a == 'P'
        x = 13
    elseif a == 'Q'
        x = 14
    elseif a == 'R'
        x = 15
    elseif a == 'S'
        x = 16
    elseif a == 'T'
        x = 17
    elseif a == 'V'
        x = 18
    elseif a == 'W'
        x = 19
    elseif a == 'Y'
        x = 20
    else
        x = 21
    end

    return x

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
