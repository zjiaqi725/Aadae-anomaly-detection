# Aadae-anomaly-detection
The AADAE for Unsupervised Anomaly Detection
## Implementation 
#### 1.Environment  
pytorch == 1.10  
torchvision == 0.4.0  
numpy == 1.21.5  
scipy == 1.4.1  
sklearn == 0.0  

#### 2.Dataset  
Publicly available real-world datasets from the ODDS repositor (http://odds.cs.stonybrook.edu)  

<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/datasets/dataset%20information.jpg" width="300" >

## Performance
### Comparison of the Receiver Operating Characteristic (ROC) curve
The experiments are conducted under two task scenarios, Task I is a weakly supervised scenario using only normal samples for training (anomaly-free), and Task II is an unsupervised scenario where the training set is randomly mixed with a few anomalies.  
<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task1.jpg" width="300" ><img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task2.jpg" width="300" >  
Task I and Task II on Pima  
<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_thyroid_task1.jpg" width="300" ><img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_thyroid_task2.jpg" width="300" >  
Task I and Task II on Thyroid 
### Special Thanks：
Our work is inspired by Gong’s work in [Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection](https://donggong1.github.io/anomdec-memae)
