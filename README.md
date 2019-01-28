# Using machine learning to estimate fail pattern in semiconductor control wafer.

The control wafer(CW) is a important subject of the cost in the semiconductor. Over the years, reclaim wafer (recycle of the CW) has been replace the new wafer for cost reduction. But there are many factor that affect the recycle successful rate. Engineers spend a lot of time to analy failure reasons. This study proposes the application of machine learning for CW recycle fail factor classification. Used EDA, statistical tools, machine learning modeling algorithms and model performance tools for creating a backend model to estimate control wafer fail pattern.


## Data Description
* Forder of `Input`: CW fail rate data. 
* Forder of `Func`: Machine learning model & parameters
* `CW1.py`: Case-1 analysis (Particle count predict)
* `CW2.py`: Case-2 analysis (Good/Bad bay predict)
* `CW1.ipynb`: Case-1 jupter notebook (include of data visualization)
* `CW2.ipynb`: Case-2 jupter notebook (include of data visualization)

## Particle count analysis & prediction
In the first case (CW1.py), I used XGBoost & LightGBM to set up out ML model. K-Fold Cross Validation is used to validate our model through generating different combinations of the data we already have. In the second case (CW2.py), I used Random Decision Forests (RF) to extract the feature and train model. The test results indicate a superior classification performance of the proposed method. 

## Evaluation Metrics
In the first case, RMSE (Root-Mean-Square Error) was used to calculate performance. In the second case, three metrics was used to calculate performance precision, recall, and F-score. The final result will be decided by F-score with β = 0.5.

## Result
The RMSE of XGBoost (RMSE=1.41) is almost equal to LightGBM (RMSE=1.40). So we choise both to analysis in the first case. In the feature importance analysis, the top five factoc is: Mainpd_id_2, EQP_ID1, EQP_ID4, EQP_ID3, Recycle, 
In the second case, the Accuracy/Precision/Recall/F-score is on testing data of RandomForestClassifier is 0.7614/0.8082
/0.8939/0.8489.

Based on the result, we find the recycle is the key feature of label. And in the previous life of CW, Route-A is singificant worse of all route. Then we go 2nd analysis, we find the step-1 eqp_id of clean process is the key feature of label. After discuess with CW sponsor, we decide to limit the recycle count and inhibition clean tool of Route-A. The second import factor is  SP recipe which  impact CW fail rate. CW sponsor had been co-work with metrology engineer to fine ture recipe.

#### For security issues, I don't provide orignal data online. It focus on methodology here.
