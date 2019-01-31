
# The "Curse of Dimensionality"

## Introduction

Since the rise of big data analytics in recent years, phrases like "Curse of dimensionality" and "Dimensionality Reduction" are becoming very popular in the areas of machine learning, deep learning and NLP etc. It does not take long before a data scientist comes across these ideas in real world analytical scenarios. In this lesson, we shall introduce curse of dimensionality, what does it entail and how can this become a serious problem as the size of our dataset grows.

## Objectives 

You will be able to:
- Describe the reasons behind the curse of dimensionality
- Explain curse of dimensionality using intuitive examples
- Describe what is meant by Sparsity and how is it related to dimensionality

*Note: Images and accompanying example used in this lesson have been taken from an excellent article by Vincent Spruyt, [The Curse of Dimensionality in classification](http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/)*

## Dimensionality and Over-Fitting 

With above scenario, imagine that there are an infinite number of cats and dogs in the world. Our dataset only contains a subset of these animals, and hence it is quite limited. Let's say we only have 10 images to start with. Our goal will be to train our classifier on these 10 training examples, in order to potentially classify (generalize) all the cats and dogs in the world. This sounds like very challenging task. 

Let's see how adding dimensions affects our "Feature Space".

### Feature Space

Feature space refers to the dimensions where our predictor variables exist. The term is used often in ML literature because a common task in ML is feature extraction, hence we view all predictor variables as potential features.

**1-D Feature Space**

To start with, we will add a single feature to our feature space. Let's pick the "average red color" as our first feature, and try to classify the data using this 1-D feature as shown below: 
![](1f.png)

So this is clearly not enough as there is some shade of red present for both classes. We can see above that in a 1 dimensional feature space, there is no clear classification boundary that can differentiate cats from dogs.

**2-D Feature Space**

How about we add another feature. We can add the "average green color" to our feature space and run the classifier again. Adding a new feature will make our feature space 2-dimensional as shown below:

![](2f.png)

So now we see that the data is much more spread out in the 2-D feature space. Still, we can not fit a linear classifier as these classes are still not linearly separable i.e. no single line through the space can differentiate between two classes. Perhaps we can add more dimensions to our feature space in an attempt to simplify the task!

**3-D Feature Space**

Let's now add in the "average blue color" as a third feature. We may see something similar to image below, where we now have a 3-D feature space.

![](3f.png)

Is the data classifiable now?. In the 3-D feature space, we can now find a plane that perfectly separates dogs from cats (Remember, this is just an example. With real world data, such classification may not be possible with three features). 

As shown below, a combination of the three chosen features can be used to obtain perfect classification results on our training data of 10 images, where cats and dogs examples reside on the opposite sides of our decision boundary.

![](plane.png)

So we see in this example scenario that __increasing the number of features until perfect classification results are obtained is the best way to train a classifier__, However, as we mentioned earlier, this is not the case. Confusing right ? Let's try to work through this  in more detail. We need to understand how sample density affects our feature space.

## Sample Density vs. Feature Space

> **The sample density is the number of recorded samples per unit distance.**

Above, we see that the density of the training samples decreases exponentially when we increase the dimensionality of the our data. It is going from being __very dense__  in 1-D feature space, to __very sparse__ in 3-D feature space. 

- In the 1-D feature space with 10 training instances covered the complete 1-D feature space having 5 unit intervals giving sample density = 10/5 = 2 samples/interval.

- In the 2-D feature space, our 10 training examples now cover a space with an area of 5x5=25 unit squares. Therefore, the sample density = 10/25 = 0.4 samples/interval. 

- In the 3-D feature space, our 10 samples cover a volume of 5x5x5 = 125 unit cubes. Therefore, the sample density is 10/125 = 0.08 samples/interval.

> __The notion of a feature space having a very low density of examples is known as "Sparsity"__.

So we see that sample density decreases (and hence the sparsity increases) with increasing dimensions. We can still choose to add more features, making our feature space even more sparse. 

With a highly sparse feature space, it becomes easier to find a separable hyperplane.

> The probability of our training data residing on the wrong side of the decision boundary becomes infinitely small when the number of features grow infinitely large. 

However, if we bring down our highly dimensional classification outcome back to a lower dimensional feature space e.g. a 2-D feature space, we see another problem as shown below:

![](sp2.png)

## Over-fitting vs. Dimensionality 

Here, we are trying to bring down the results of 3-D feature space classification to the 2-D feature space. We saw that our data was successfully classified in 3-D space, but this classification can not be shown in the 2-D space with a linear classifier. The main reason behind this is that adding new dimensions allows the classifier to perform complicated non-linear classification in order to learn the given 10 instances in great details. So, with the given set of dimensions and number of examples, the classifier will perform well on training data but will fail to generalize for previously unseen data. As we mentioned, considering an infinite number of cats and dogs in the world, our results are very limited, highly over-fitted and the classifier is not suitable for any predictive analysis. 

This over-fitting is due to the curse of dimensionality. The following figure shows the result of a typical linear classifier that operates in a 2-D feature space. 
![](2dok.png)

Although this is not perfect classification as we had in 3-D feature space, this not-so-perfect classifier will be more __generalizable__ for previously unseen data. In other words, by using less features, the curse of dimensionality can be avoided such that our classifier does not over-fit training data. 

## Sparsity with Continuous Features

Let's now imagine that we want to train a classifier using only a single unique continuous feature with value ranging from 0 to 1. This could be any normalized feature i.e. height or weight of cats and dogs. 

![](cont.png)

- If we want our training data to cover 20% of the predictor range (left image), the amount of training data needed is 20% of the complete population of cats and dogs. 

- If we add another feature, resulting in a 2-D feature space (middle image) to cover 20% of the 2-D feature range, we now need to obtain 45% of the complete population of cats and dogs in each dimension (0.45^2 = 0.2). 

- For 3-D feature space (right image), in order to cover 20% of the range, we need to obtain 58% of the population in each dimension (0.58^3 = 0.2).

### So what do we conclude from above ?

> If the amount of available training data is fixed, then over-fitting occurs if we keep adding dimensions.

> If we keep adding dimensions, the amount of training data needs to grow in order to to avoid over-fitting.

This can be summarized as :
__Dimensionality Causes Sparsity__

## Non-uniform Sparsity

Another undesirable effect of the curse of dimensionality is that this sparsity in high dimensional space is not uniformly distributed. Data around the origin is much more sparse than data in the corners of the space. 

Consider a unit square representing 2-D feature space as shown below, having the data points within a unit circle that fits in this unit square. 

![](unit.png)


All the examples outside this unit circle are closer to the corners of the given 2-D space, than to its center. 
This makes these examples hard to classify as their feature values differ by a huge margin. If most samples fall inside the unit circle, our classification task becomes much simpler. 

As we keep on adding more dimensions, we can think of our unit square becoming a unit cube and eventually a **hypercube**. Following image shows a 2-D unit square, a 3-D unit cube, and an imaginary visualization of an 8-D hypercube with 2^8 = 256 corners.
![](8d.png)

>  **In geometry, a hypercube is an n-dimensional analogue of a square (two dimensions) and a cube (three dimensions)** 

In the 8-D hypercube, roughly 98% of the data is present in its 256 corners. This shows that as the dimensionality of the feature space goes to infinity, the ratio of the difference in minimum and maximum distance from sample point to the center tends to zero. Hence the classification becomes an almost impossible task.

### Dimensionality and Modeling Challenges

Most distance measures (e.g. Euclidean, Mahalanobis, Manhattan etc.) are less effective in highly dimensional spaces as due to sparsity, the distance between examples slowly becomes meaningless as we continue to increase dimensions. We also saw that Classification is often easier in lower-dimensional spaces where less features are used to describe the object of interest. Similarly, Gaussian likelihoods become flat and long tailed distributions in high dimensional spaces.


Next, we will look at the techniques available to deal with the curse of dimensionality. We will look at some machine learning and feature extraction techniques to bring high dimensional data into a low dimensional space. 

## Summary

In this lesson, we looked at an introduction to curse of dimensionality and saw how increasing dimensions. without increasing the number of training examples can lead to poor classification and prediction. Next, we will look at ways to avoid such a scenario and options we have available to us.

## Additional Resources

You are encouraged to refer to following documents for further details/examples on curse of dimensionality 

- [Surprises in High Dimensions, Hypersphere and Hypercube](http://www.maths.manchester.ac.uk/~mlotz/teaching/suprises.pdf)
- [Machine Learning: Curse of Dimensionality](https://www.edupristine.com/blog/curse-dimensionality)
- [Curse of Dimensionality (in a regression context)](https://shapeofdata.wordpress.com/2013/04/02/the-curse-of-dimensionality/)

