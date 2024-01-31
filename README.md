# Latent Topic Block Model
Project as part of the "AI Methods" course taught by [Christine Keribin](https://wp.imo.universite-paris-saclay.fr/christine-keribin/) in the second year of the [Mathematics and Artificial Intelligence master](https://www.imo.universite-paris-saclay.fr/fr/etudiants/masters/mathematiques-et-applications/m2/m2-mathematique-et-intelligence-artificielle/)'s program. 

Study of residual networks based on the paper [The latent topic block model for the co-clustering of textual
interaction data](https://www.sciencedirect.com/science/article/pii/S0167947319300726?via%3Dihub) by Laurent R. Bergé, Charles Bouveyron, Marco Corneli, Pierre Latouche (2019).

<p align="center">
<img src="https://github.com/pierrecavalier/M2_NSA/blob/main/pictures/LTBM.png" width="300">
  </p>

## Experiment

We have implemented the latent topic block model by generating our sample (like the first example in paper) by representing word with a unique integer and by creating sentence with integer from a cluster and we printed the result with the interaction between the two cluster.

<p align="center">
<img src="https://github.com/pierrecavalier/M2_NSA/blob/main/pictures/Result.png" width="600">

</p>


## Usage

You can use the notebook clean.ipynb to generate random data and try the co-clustering. Unfortunatly the code is not optimized and therefore, it would be better to vectorize it for better performances.
