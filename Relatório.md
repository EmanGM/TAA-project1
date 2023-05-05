# Abstract 

# Introduction


More and more we see an increased interest in automated vehicle driving both in the manufacturers as well as in common citizens wanting to buy such products. The relevance of this study is ...
This consists of classifying traffic signs according to one of these 43 labels

![imagem](./images/all43classes.png)


# Why we chose this project

The world of automation in cars is fascinating to us and it's becoming increasingly evident that a car's software is just as critical as its physical and mechanical components. As a result, it's highly probable that many of us may have the opportunity to work in this field in the future.
how many more people dedicate themselves to this subject of study, more quickly and easier it will be to find a robust and very efficient model which is able to correctly classify the traffic signs. Although our contribution may not be  This is a very relevant topic because ...

# Data visualization



# Methods

As we can't change data, we need to develop this project in a model-centric view.
(As data are in pixel gradient from 0 to 255 there is no need for normalization.)
Due to class imbalance, accuracy is not a good metric, so we are using F1 score.
We decided to leave convolution neural networks for the next project 

# Models 

## Logistic Regression

## Artificial Neural Network

## K-nearest Neighbors

This method is very simple and is used in classification problems. It checks the neighbors of a given point in an n-dimension space and by comparing the distances to other points the most *k* near examples vote and the class that has more votes wins.  
This seems a good method but it doesn't learn anything, it does not have the ability to improve itself over time. Its computational time scales linearly $O(N)$ as it has to compare each test sample to every data point in the training set making it inefficient with large datasets. Also, there's no deterministic method to choose the best *k* so a try-and-guess approach is needed.  
This method is good when the number of features is low, however, our case is the opposite, so we expect a low performance using this approach. 
To run this method, all the images are in greyscale and resized to 30x30.

<br/>

![finding k](./images/k-nearest%20neighbors.png)   

![average_accuracy](./images/k-nearest%20neighbors%20accuracy.png)

As we can see from the graph, K-nearest Neighbors doesn't yield good results with any value of *k*, in fact no more than 37% of F1 score was possible to achieve. These results were according to our expectations. 

## SVM




# Results

# Conclusions

(indicar coisas que poderiam ser melhoradas)

# References 

ver esta função do sklearn
gridSearchCV()