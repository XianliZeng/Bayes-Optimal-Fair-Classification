# Bayes-Optimal-Fair-Classification
Codebase and Experiments for Bayes-Optimal Fair Classification  with Linear Disparity Constraints via Pre-, In-, and Post-processing

# Codebase Overview
This repository provide 13 different algorithms for fair classifications. The implementations of Fair Up- and Down-Sampling (FUDS), Fair Cost-Sensitive Classification (FCSC), Fair Plug-In Rule (FPIR), and other benchmark methods are located in the 'Algorithms' folder. To replicate the real data analysis from our paper "Bayes-Optimal Fair Classification with Linear Disparity Constraints via Pre-, In-, and Post-processing," please execute main.py. This script will handle the training, testing, and result-saving processes for each method across various datasets. Additionally, by running the print_table and draw_plot functions, you can generate the simulation results that are presented in our paper.

# Algorithms Considered
This repository provide python realization of 13 benchmark methods of fair classification,  comprising four pre-processing, five in-processing, and four post-processing methods each.

-pre-processing: 

--Fair Up- and Down-Sampling (FUDS): X. Zeng, K. Jiang, G. Cheng, and E. Dobriban. Bayes-Optimal Fair Classification  with Linear Disparity Constraints via Pre-, In-, and Post-processing.

--Disparate Impact Remover (DIR): 
  M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and S. Venkatasubramanian. Certifying and removing disparate impact.
  
--FAWOS:
  T. Salazar, M. S. Santos, H. Ara ́ujo, and P. H. Abreu. FAWOS: Fairness-aware oversampling algorithm based on distributions of sensitive attributes.

--Adaptive Reweighing Algorithm (ARA):
  J. Chai, X. Wang. Fairness with Adaptive Weights.
  
-in-processing:

--Fair Cost-Sensitive Classification (FCSC): X. Zeng, K. Jiang, G. Cheng, and E. Dobriban. Bayes-Optimal Fair Classification  with Linear Disparity Constraints via Pre-, In-, and Post-processing.

--KDE based constrained optimization (KDE)：
  J. Cho, G. Hwang, and C. Suh. A fair classifier using kernel density estimation.
  
--Adversarial Debiasing (ADV):
  B. H. Zhang, B. Lemoine, and M. Mitchell. Mitigating unwanted biases with adversarial learning.

--MinDiff (MD):
  F. Prost, H. Qian, Q. Chen, E. Chi, J. Chen, A. Beutel. Toward a better trade-off between performance and fairness with kernel-based distribution matching.
  
--Reductions (RED):
  A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, H. Wallach. A Reductions Approach to Fair Classification.
-post-processing

-post-processing:

--Fair Plug-In Rule (FPIR): 
  X. Zeng, K. Jiang, G. Cheng, and E. Dobriban. Bayes-Optimal Fair Classification  with Linear Disparity Constraints via      Pre-, In-, and Post-processing.
  
--Post-processing through Flipping (FFP)
  W. Chen, Y. Klochkov, and Y. Liu. Post-hoc bias scoring is optimal for fair classification.
  
--Post-processing through Optimal Transport (PPOT)
  R. Xian, L. Yin, and H. Zhao. Fair and optimal classification via post-processing.

--Group Fairness Framework for Post-processing Everything (FRAPPE)
  A. Tifrea, P. Lahoti, B. Packer, Y. Halpern, A. Beirami, F. Prost. FRAPPE: A Group Fairness Framework for Post-Processing Everything.

# Related Works and Repositories
This repository has draw lessons from other open resourses. 

--Codes for DIR, ADV take inspiration from the AI Fairness 360 platform:  https://github.com/Trusted-AI/AIF360;

--Codes for FAWOS take inspiration from: https://github.com/teresalazar13/FAWOS; 

--Codes for KDE follows the original code provided by: J. Cho, G. Hwang, and C. Suh. A fair classifier using kernel density estimation;

--Codes for PPOT take inspiration from: https://github.com/rxian/fair-classification;

--Codes for PPF take inspiration from the paper:   W. Chen, Y. Klochkov, and Y. Liu. Post-hoc bias scoring is optimal for fair classification.

--Codes for FRAPPE take inspiration from: https://github.com/google-research/google-research/tree/master/postproc_fairness

# Data
This repository uses the AdultCensus, COMPAS, Law School, and ACSIncome datasets. They can be found in the Datasets folder and are loaded using dataloader.py.
