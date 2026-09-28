# Flight-Cancellation-Prediction-Model

This is a machine learning project I built to predict whether a flight will be cancelled or not. The dataset is 100,000 rows.



* Platform: Databricks (Serverless Compute Node)
* Language: Python (Pandas, NumPy, Sklearn, Skopt)
* ML Model: LightGBM Classifier (`LGBMClassifier`)



### Phase 1: Raw Logistic Regression
* Built an initial model using raw data and only three features.
* Performed terribly, landing a score of 0.52 ROC-AUC which was basically random guessing

### Phase 2: Eliminating Data Leakage
* As I engineered more historical averages, the model suffered from data leakage by accidentally pulling metrics from future test horizons to predict past events.
* I sorted the data chronologically by ['FL_DATE', 'CRS_DEP_TIME'] and forced train_test_split to turn`shuffle=False. 
* This forced the model to strictly predict the future based on the past, making the benchmark realistic and much tougher.
* During this phase, I discovered that ROC-AUC requires raw probability outputs .predict_proba() rather than binary predictions .predict() to calculate risk rankings accurately. Leveraging this insight boosted our benchmark score to 0.58.

### Phase 4: LightGBM 
* Transferred the pipeline over to LightGBM.
* The raw tree model initially showed almost no performance increase.

### Phase 5: Bayesian Optimization Search
* I deployed a BayesSearchCV loop across 25 iterations to tune key tree parameters: Alpha, Lambda, learning_rate, max_depth and more.
* Running initial optimization allowed the trees to jump to 0.685.
* I increased the parameters range by tuning it even more as some parameters such as alpha and lambda had maxed out.


## 📊 Final Performance Results

* [[15785  3805]      # 15785 flights predicted as on time, 3805 flights predicted as risk but flew.
* [  214   196]]     # 196 flights caught as cancelled, 214 flights missed. (ended up getting cancelled by model didnt catch it)
* ROC-AUC 0.694729765061816
* 0.0793480689300071

   
## What I Learned from this Project
I learned that when a dataset (like this one) is so skewed to one side (not cancelled), it becomes extremely difficult to predict such as rare occurance such as cancellation, that the model doesnt have enough information to be much more accurate , forcing the model to give out false positives and miss more flights purely due to lack of data instead of insufficent features. It is much harder to predict than a variable such as load factor as every flight would have a load factor and there would be much more data available such as capacity and seats sold.
