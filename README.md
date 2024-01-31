# ResNet
Project as part of the "AI Methods" course taught by [Christine Keribin](https://wp.imo.universite-paris-saclay.fr/christine-keribin/) in the second year of the [Mathematics and Artificial Intelligence master](https://www.imo.universite-paris-saclay.fr/fr/etudiants/masters/mathematiques-et-applications/m2/m2-mathematique-et-intelligence-artificielle/)'s program. 

Study of residual networks based on the paper [The latent topic block model for the co-clustering of textual
interaction data](https://www.sciencedirect.com/science/article/pii/S0167947319300726?via%3Dihub) by Laurent R. Bergé, Charles Bouveyron, Marco Corneli,
Pierre Latouche (2015).

<p align="center">
<img src="https://github.com/pierrecavalier/ResNet/blob/main/docs/resnet.png" width="300">
  </p>

## Experiment

We have implemented residual networks with a number of layers ranging from ten to fifty, trying out two options (A and B, described in the article) and determining their accuracy on a subset of the CIFAR-10 dataset.

<p align="center">
<img src="https://github.com/pierrecavalier/ResNet/blob/main/results/OptionAandB.png" width="300">
<img src=https://github.com/pierrecavalier/ResNet/blob/main/results/ResNetandTorchResNet.png width="300">
</p>


## Usage


Use "streamlit run streamlit.py" to run the website.

One can vizualise, train the chosen model by clicking respectively on the buttons "View model" and "Train model".

The training of the model is done on a subset of 5000 data of CIFAR10 on 5 epochs which should take about 3-5 minutes.

Once all the desired model trained, one can compare their accuracy by clicking on the last button, "Print the accuracy of every model trained so far".
