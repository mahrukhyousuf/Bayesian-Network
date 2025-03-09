# **Predicting Student Academic Success**  - Probablistic Graphical Modelling

## **Project Overview**  
This project develops a **Bayesian Network (BN)** model for my course **Probabilistic Graphical Modelling** analyze and predict factors influencing **student academic success**. The model incorporates key variables like **attendance, motivation, teacher quality, and hours studied**, providing probabilistic predictions for student outcomes (**Fail, Pass, Excellent**).  

## **Why Bayesian Networks?**  
- Academic success is influenced by **uncertain** and **interdependent** factors.  
- BNs efficiently model **causal relationships** and **uncertainties** in real-world data.  
- The model integrates **expert knowledge** and **data-driven learning** for robust predictions.

## Files:
- IEEE Research Paper - pgm.pdf
- k2.png
- domainexp.png

# Programming Language Use:
Python

## **Methodology**  
🔹 **Model Development:** Bayesian Network structure with causal links (e.g., **Teacher Quality → Motivation → Exam Score**).  
🔹 **Parameter Learning:** Trained on **6,000+ records** (Kaggle dataset) using **Maximum Likelihood Estimation (MLE)** with **Laplacian smoothing**.  
🔹 **Inference:** Predicts **Exam Scores** based on evidence (e.g., **low study hours, high teacher quality**).  
🔹 **Evaluation Metrics:** **Accuracy, Precision, Recall, and F1-score** for Fail, Pass, and Excellent categories.  

## **Structure Learning Approaches**  
### **1. K2 Algorithm (Data-Driven Structure)**  
❌ Produced **illogical relationships** (e.g., **Hours Studied → Distance from Home**).  
❌ Failed to recognize the **Excellent** class.  
✅ **Accuracy:** **88.53%**  
✅ Strong **Pass class** predictions (F1-score **93%**) but weak for **Fail** cases.  

### **2. Expert-Driven Structure (Domain Knowledge-Based)**  
✅ More **realistic causal relationships**, verified by **domain experts**.  
✅ Improved prediction for **Fail & Excellent categories**.  
✅ **Accuracy:** **84.99%** (comparable to K2 but with better minority class predictions).  

## **Key Findings**  
✔ The **expert-driven model** is more interpretable and **practically meaningful**.  
✔ While the **data-driven model** had higher accuracy, it **misrepresented causal links**.  
✔ The **expert-driven approach** performs better in handling **minority classes** (**Fail, Excellent**).  

## **Conclusion**  
A **hybrid approach** combining **data-driven learning** and **expert knowledge** can significantly improve predictive accuracy while ensuring **logical, real-world relevance**.  
