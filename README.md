# Stock Market Tweet Analysis

### Source Code: https://github.com/kasamdh/CourseProject/blob/main/StockMarketSentimentAnalysis.ipynb
### Video Tutorial/Presentation: https://youtu.be/Lpwx3a3o3X0
### Data: https://github.com/kasamdh/CourseProject/blob/main/stock_data.csv
### Powerpoint Presentation: https://github.com/kasamdh/CourseProject/blob/main/CS%20410%20%E2%80%93%20Final%20Presentation.pptx
### Final Project Documentation:https://github.com/kasamdh/CourseProject/blob/main/Final_Project_Documentation.pdf

<h1>Team Members</h1>
      
* Kasam Dhakal (kdhakal2@illinois.edu) - Captain <br/>

* Nisarg Mistry (nmistry2@illinois.edu)<br/>

* Parth Shukla (pshukl21@illinois.edu )<br/>

<h1>Goal</h1>
<p>Capture sentiment analysis, collect Twitter sentiment towards the stock market.</p>


<h1> Introduction </h1>
<p1> Numerous factors, including both internal and external factors, can affect and move the stock market. Various data mining techniques are frequently employed to address this problem. Stock prices change instantly due to shifts in supply and demand. On the other hand, machine learning will offer a more precise, accurate, and logical method for addressing stock and market price concerns. A novel approach to creating simulation models that can predict stock market movements and whether they will increase, or decrease has been improved using ML algorithms. Several sentiment analysis studies used Support vector machines (SVM), Naive Bayes regression, Random Forest Classifier, and other techniques. The effectiveness of machine learning algorithms depends on the quantity of training data available.  To begin, we train many algorithms using Sentiment 5792 clean Twitter data. We utilized SVM to ascertain the typical sentiment of tweets for each trade day because this was the emotional analysis that performed the best. </p1>

<h1>Problem Statement </h1>
<p1> In this project, we attempt to put into practice an NLP Twitter sentiment analysis model that aids in overcoming the difficulties associated with determining the sentiments of the tweets. The following information is necessary for the dataset used in the twitter sentiment analysis project:
The Sentiment Dataset, which was made available (https://www.kaggle.com/datasets/utkarshxy/stockmarkettweets-lexicon-data for Sentiment Analysis), consists of 5972 cleaned tweets that were retrieved using the Twitter API. The dataset's numerous columns include:
      
1.Text: Twitter Data <br/>
2.Sentiment: Labeled data corpus for Sentiment AnalysisSentimentIntensityAnalyzer: library for classifying of sentimentsResults exported in text files.                 Probability of positive and negative Compound values in a range of -1 to 1, where -1 represents negative for each tweet. <br/> </p1>

<h1>Tools, System and Dataset </h1>

 * Google Collab <br/>

 * http://www.tweepy.org/ - Python Library to access the Twitter API  <br/>

 * http://www.nltk.org/ - Natural Language Toolkit <br/>
  
 * Twitter data from Kaggle: https://www.kaggle.com/datasets/utkarshxy/stockmarkettweets-lexicon-data for Sentiment Analysis <br/>

<h1> Instructions on how to install the software <h1/>
      
For this project we have used Google Collab
1.	Login to the google account. Create a new account if you don’t have already.
2.	Download the source code from GitHub.
https://github.com/kasamdh/CourseProject/blob/main/StockMarketSentimentAnalysis.ipynb
3. Upload the source code to Collab (http://colab.research.google.com/)
Steps: From http://colab.research.google.com/
Select: File ->Upload Notebook->Upload->Choose File

![image](https://user-images.githubusercontent.com/22782181/206098930-94f1ab13-a1a4-4675-8710-eda33a541336.png)
      
4. Dataset Connection to Connect to Google Drive: 
(https://github.com/kasamdh/CourseProject/blob/main/stock_data.csv)
Steps: Download the stock_data.csv from GitHub and upload data set to the Google Drive.
File path : Copy the file path from the Google Drive and replace it to 	DATA+DIR = mypath + “ ” in the Initilized relevant data URI’s section in 	Google Collab

![image](https://user-images.githubusercontent.com/22782181/206099191-6baef16f-44eb-44ca-836d-d6a91c447280.png)
      
 <h1>Process flow diagram</h1>
      
 ![image](https://user-images.githubusercontent.com/22782181/206099691-a5372c7b-638f-443c-ae18-b986df382242.png) 
      
      
<h1> Processing</h1>
<h3>Data Collection</h3>
<p1> Tweet data from https://www.kaggle.com/datasets/utkarshxy/stockmarkettweets-lexicon-data for Sentiment Analysis.</p1>
      
<h3>Data Cleaning</h3>
<p1>Import Stopwords corpus for cleaning the tweets, split data into train, test.</p1>
      
<h3>Vectorization</h3>
<p>1There are set of techniques use for extracting meaningful information from text corpus. Which is word vectorization. A word in a vector in the text corpus. </p1>
      
<h3>TF-IDF Transformer</h3>
<p1> The vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus.</p1>
      
<h1> Analysis</h1>
      <h3>Sentiment Analysis and Data Visualization</h3>
<p1>Labeled data corpus for Sentiment Analysis SentimentIntensityAnalyzer: library for classifying of sentimentsResults exported in text files. Outputs: Probability of positive and negative. Compound values in a range of -1 to 1, where -1 represents negative for each tweet. The compound value is comparable to a single measure of polarity. </p1>
      
<h1> Results</h1>
      <h3>Logistic Regression</h3>
      <p1>Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes. In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no). Logistic regression model predicts P(Y=1) as a function of X.</p1>
      <h3>Support Vector Machine (SVM)</h3>
      <p1>Support Vector Machine: An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. The hyperplane will be generated in an iterative manner by SVM so that the error can be minimized. The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH).</br>
The followings are important concepts in SVM − </br>
1.	Support Vectors − Datapoints that are closest to the hyperplane is called support vectors. Separating line will be defined with the help of these data points.</br>
2.	Hyperplane − As we can see in the above diagram, it is a decision plane or space which is divided between a set of objects having different classes.</br>
3.	Margin − It may be defined as the gap between two lines on the closet data points of different classes. It can be calculated as the perpendicular distance from the line to the support vectors. Large margin is considered as a good margin and small margin is considered as a bad margin</br>
</p1>
      <h3>Naive Bayes Classifier</h3>
      <p1>Naïve Bayes algorithms is a classification technique based on applying Bayes’ theorem with a strong assumption that all the predictors are independent to each other. In simple words, the assumption is that the presence of a feature in a class is independent to the presence of any other feature in the same class. In Bayesian classification, the main interest is to find the posterior probabilities</p1>
      <h3>Stochastic Gradient Descent Classifier</h3>
      <p1>This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate)</p1>
      <h3>K Nearest Neighbors Classifier</h3>
      <p1>The k-nearest neighbors' algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.</p1>
      <h3>Random Forest Classifier:</h3>
      <p1>Random forest is a supervised learning algorithm which is used for both classification as well as regression. But however, it is mainly used for classification problems. As we know that a forest is made up of trees and more trees means more robust forest. Similarly, random forest algorithm creates decision trees on data samples and then gets the prediction from each of them and finally selects the best solution by means of voting. It is an ensemble method which is better than a single decision tree because it reduces the over-fitting by averaging the result.</p1>
      <h3>Gradient Boosting Classifier</h3>
      <p1>This algorithm builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary classification is a special case where only a single regression tree is induced.</p1>
      <h3>XGBoost Classifier</h3>
      <p1>XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.</p1>
      
 <h1> Test</h1>
      <h3>Classifier Test</h3>
     1. Classifier on input </br>
     2.  Multinomial Naive Bayes Classifier</br>


      
      

      
  


      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
    
      
      
      
      

      
    





