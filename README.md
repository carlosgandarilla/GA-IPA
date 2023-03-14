# Combining GA and IPA to infer paralog pairing
In order to predict interaction partners among paralogs from two protein families, just from their sequences, we will use successively two distinct methods. The first one, Graph Alignment ([GA](https://doi.org/10.48550/arXiv.0905.1893)) algorithm, exploits phylogenetic relationships between interaction partners, via sequence similarity (i.e., it aims at identifying interologs). The second, Iterative Pairing Algorithm (IPA), relies on the inter-family coevolutionary signal as detectable by Direct Coupling Analysis ([DCA-IPA](https://doi.org/10.1073/pnas.1606762113)) and Mutual Information ([MI-IPA](https://doi.org/10.1371/journal.pcbi.1006401)).

Both signals can be used to computationally predict which proteins are specific interaction partners among the paralogs of two interacting protein families, starting just from their sequences. We show that combining them improves the performance of interaction partner inference, especially when the average number of potential partners is large and when the total data set size is modest.

Paper: "Combining phylogeny and coevolution improves the inference of iteration partners among paralogous proteins"(link to the paper) (Carlos A. Gandarilla-Perez, Sergio Pinilla, Anne-Florence Bitbol and Martin Weigt xxxxxxx , 2022)
<img width="1742" alt="Figure1" src="https://user-images.githubusercontent.com/43339953/191007155-de52cfd7-fdd1-4332-b891-f0743a5d755c.png">
We provide here the code to reproduce the key results and figures of the paper.

## Usage:

To run the code, you first need to install [julia](https://julialang.org/) and then run the following commands to test the demo:
```
include("./GA-IPA_mai.jl")
n_replicates = 100
GA_IPA_DCA_robust(n_replicates) or GA_IPA_MI_robust(n_replicates)
```

where n_replicates is the number of realizations of GA experiment, each time taking different random matchings as starting points.

Here it is applied to a dataset of 5052 sequences of cognate histidine kinases and response regulators from the [P2CS](http://www.p2cs.org/) database. To run the code on the other datasets, unzip data from CoAlignments.zip, open the file src/GA_IPA_DCA_robust.jl or src/GA_IPA_DCA_robust.jl with your favorite editor, and replace protfile with the new protein name. 

[![DOI](https://zenodo.org/badge/527516761.svg)](https://zenodo.org/badge/latestdoi/527516761)
