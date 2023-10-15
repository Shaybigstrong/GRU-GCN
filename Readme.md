Spatiotemporal prediction of carbon emissions using a hybrid deep learning model considering temporal and spatial correlations 

Task
This paper presents a deep learning-based hybrid prediction model for carbon emissions. The model enables the prediction of future carbon emissions, in single- and multi-step scenarios, by using historical time series data. The monthly ODIAC data for the three major urban agglomerations in China, namely, the Yangtze River Delta, Pearl River Delta and Beijing–Tianjin–Hebei, were utilised for performance evaluation.

Environment
	CPU: i5-11400H @ 2.70 GHz,
	GPU: NVIDIA GeForce RTX3050 4 GB
	RAM: 16 G. 
	Encoding language：Python 3.7.12

Dataset
The ODIAC datasets used in this study are available at http://www.odiac.org/index.html
Preprocessed data：
Yangtze River Delta：
GYH62.xlsx is the preprocessed feature matrix
LJJZ.xlsx is the adjacency matrix
JLJZ.xlsx is the distance weight matrix
Pearl River Delta：
GYH62.xlsx is the preprocessed feature matrix
LJJZ.xlsx is the adjacency matrix
JLJZ.xlsx is the distance weight matrix
Beijing-Tianjin-Hebei：
GYH62.xlsx is the preprocessed feature matrix
LJJZ.xlsx is the adjacency matrix
JLJZ.xlsx is the distance weight matrix
Example of multi-step prediction：prediction4.xlsx

Model
GRU-GCN.ipynb
Two versions of the model trained on the Yangtze River Delta dataset using different weight matrices: CSJ.pth and CSJ2.pth
