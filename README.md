
# Machine learning model  
  
This exercise is based on a simplified version of a real model that we have built on the team.  
  
We want to predict whether a customer is going to return (order again from us) or not in the next 6 months.  
  
Two CSV files are provided: one with [past order data](./data/machine_learning_challenge_order_data.csv.gz) for a sample of customers, and another with the [labels](./data/machine_learning_challenge_labeled_data.csv.gz) for the same set of customers.  
  

## Description of the problem & tips

  
In this simplified example, the idea is to create an algorithm that, for each customer, and by considering only the order history provided, predicts correctly the label in the labeled data (true / false prediction).  
  
The order data is composed of a variable number of input rows for each customer. Your mission is to prepare the data and use it to train a machine learning algorithm.  
  

## Answers to provide

  
You can solve this exercise using the Python data ecosystem with the usual well-known tools and libraries.  
  
Please send back to us:  

 - all the code and files you used (both for exploration and the final ones) as a GIT repository (see below for further details)  
 - a summary of the data transformations that you tried, and the ones you kept (we should be able to see this in the commit history)  
 - a description of the ML models you tried, and the one you kept (we should be able to see this in the commit history)  
 - a measurement of how good your models are (at least for the best one)  
 - any other findings on the data that you would like to share  
 - ideas about any further models or data transformations that you would try if you had more time  
  
Please, make commits often, as we want to see the thought process that you followed during the exercise.  
  
### How to submit the code  
  
Please clone this repo to a public Github repo that does not contain an obvious name (to avoid other candidates from copying it). Just choose a couple of random dictionary words and name your repo with them :-)  
  
The GitHub repo should contain:  
 - all the code (both for exploration and the models) and the commits' history showing the process you followed  
 - the instructions to train the model you choose and evaluate it  
  
Please, share *all* the code you write, in case we have further questions.  
  
### How we will evaluate the task  
Do the exercise it at your own pace, and take into account that we value:  
 - Structure of the repository  
 - The simplicity of the code (using well-known libraries to achieve this is allowed and encouraged)  
 - Readability  
 - Correctness  
 - That the commits are atomic and meaningful  
 - In general, the use of good development practices and Python principles and idioms  
 - We value having the code delivered in a way that it will be easy to run from a freshly-cloned repository and that would be easy to put in production  
  
Regarding the modeling task, we value  
 - Exploration  
 - Feature engineering  
 - ML algorithm (or algorithms) that you applied and how you applied them  
 - Evaluation  
  
## Final notes  
In our evaluation we will not weigh too much how accurate your model is, because we know it would require a larger amount of time to get very good accuracy, but rather the process you followed, as detailed in the questions 1 to 6 above. We will evaluate particularly if the choices of data transformations and models make sense, your ideas for further improvements, and the quality of the code.  
  
We hope you'll find this exercise interesting and challenging.   
Of course, we are happy to answer any questions to clarify the problem. Please feel free to drop us as many emails as needed.  
  
Thank you very much for your time!  
  
  
## Data dictionary  

### Order data  
The [order_data CSV](./data/machine_learning_challenge_order_data.csv.gz) contains the full order history of a sample of customers that were acquired at or after 2015-03-01. The provided order history ends on 2017-02-28. So there is order data only between 2015-03-01 and 2017-02-28.  
The data is synthetic to prevent data leakages, but it behaves like real data for the purposes of this exercise.  
  
These are the columns of the order data CSV file:  

 - *customer_id* ⇒ Uniquely identifies a single customer  
 - *order_date* ⇒ Local date of the order  
 - *order_hour* ⇒ Local hour of the order  
 - *customer_order_rank* ⇒ Successful orders for a particular customer get numbered in chronological order (starting on 1) in this column. The failed orders have this column empty  
 - *is_failed* ⇒ **0** if the order succeeded. **1** if the order failed (and then *customer_order_rank* will be empty)  
 - *voucher_amount* ⇒ If the customer used a voucher (discount) for this order, the discounted amount is put here  
 - *delivery_fee* ⇒ Fee charged for the delivery of the order (if applicable)  
 - *amount_paid* ⇒ Total amount that the customer paid (the *voucher_amount* is already deducted and the *delivery_fee* is already added in the *amount_paid* column)  
 - *restaurant_id* ⇒ Identifies a restaurant  
 - *city_id* ⇒ Identifies a city  
 - *payment_id* ⇒ Identifies the payment method that the customer chose (such as cash, credit card, paypal, ...)  
 - *platform_id* ⇒ Identifies the platform the customer used to place the order (web, mobile app, mobile web, …)  
 - *transmission_id* ⇒ Identifies the method that was used to send the order to the restaurant (fax, email, phone, and different kinds of proprietary devices or point-of-sale systems)  
  
For convenience, the data is ordered by *customer_id* and then by *order_date*, so all the orders of the same customer appear in consecutive rows and in chronological order.  
It is not necessary to know the exact meaning of the IDs, therefore we do not provide the mapping from the IDs to actual restaurants / cities / payment methods / platforms / transmission methods.  

## Labeled data  
The [labeled_data CSV](./data/machine_learning_challenge_labeled_data.csv.gz) contains the true/false information about whether the customers did order in the 6 months after 2017-02-28 or not.  
  
These are the columns of the order data CSV file:  

 - *customer_id* ⇒ Uniquely identifies a single customer  
 - *is_returning_customer* ⇒ **0** if the customer did not return (did not order again in the 6 months after 2017-02-28). **1** if the customer returned (ordered again at least once after 2017-02-28)  
  
For convenience, the data is ordered by *customer_id*.
