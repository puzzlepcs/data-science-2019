# Data Science
Repo for 2019, 1st semester, Data Science (ITE4005) class.  
## Tested Envirionment
- OS: Windows 10
- Language: Python 3.6.8

## Assignment 01: Apriori algorithm
### Summary of Algorithm
In order to get all the association rules, apriori algorithm was used to mine all frequent itemsets. All support value data of candidate itemset were saved during the process.  
Next, association rules were generated and following confidence value were calculated using frequent itemsets and support value data mined from apriori algorithm.  

### Compilation
```python apriori.py [minimun support(%)] [input file name] [output file name]```


## Assignment 02: Decision Tree 
### Summary of Algorithm
In order to build tree easily, dictionary type was used to represent a tuple. Then, class attribute was paired with each tuple in training dataset. 
I implemented decision tree which uses information gain or gain ratio and both values were tested after implementation. 
Using gain ratio split subsets more evenly than using information gain since it is normalized version of information gain. Also, when tested with given data, gain ration turned out to perform better. Lastly, gini index was not appropriate for the given datasets since gini index is used only for binary splits. 
For these reasons, gain ratio was chosen as the criteria for choosing best attribute.

### Compilation
```python dt.py [train data file name] [test data file name] [output file name]```


## Assignment 03: DBSCAN(density-based spatial clustering of applications with noise)
### Summary of Algorithm
Given a set of points in some space to be clustered, the points are classified as core points, (density-) reachable points and outliers.
- A point p is a core point if at least minPts points are within distance eps of it.
- A point q is directly reachable from p if point q is within distance eps from the core point p. 
- A point q is reachable from p if there is a path p_1, …, p_n with p_1=q, where each p_(i+1) is directly reachable from p_i. (This implies that all points on the path must be core points, with the possible exception of q.)
- All points not reachable from any other point are outliers.
Now if p is a core point, then it forms a cluster together with all points (core and non-core) that are reachable from it.   

Starting with an arbitrary point P that has not been visited, P’s neighborhood is retrieved, and if it contains sufficiently many points, a cluster is started. Otherwise, the point is labeled as an outlier.  
If a cluster is started, it begins to grow, considering reachable points as its member. Process that grows certain cluster are done only with the points that are not labeled. The process continues until the density-connected cluster is completely found. Then, a new unvisited point is retrieved and processed, leading to the discovery of a further cluster or noise.

### Compilation
```python clustering.py [input file name] [n] [eps] [MinPts]```

## Term Project: predicting ratings of movies
### Collaborative Filtering
#### Overview
- Memory-based methods: Doesn't need anything else except user's historical preference on a set of items. Doesn't make any models to predict ratings.
    - user-based
    - item-based
    - Pros
        - Easy to implement
        - Easy to understand
    - Cons
        - Sparse data
        - Not appropriate for realtime recommendations.
- Model-based: based on matrix factorization. Better at dealing with sparsity
    - examples: decision trees, rule-based models, Bayesian methods, latent factor models


#### User-based Collaborative Filtering
The standard method of collaborative filtering is known as Nearest Neighborhood algorithm. There are user-based CF and item-based CF. We have n x m matrix of ratings, with user u_i, i = 1, ..., n and item p_j, j = 1, ... , m. __Now we want to predict the rating r_ij if target user i did not watch/rate an item j.__   

The process is to calculate the similarities between target user i and all other users, select the top X users, and take the weighted average of ratings from these X users with similarities as weights.  

Different people may have different baselines when giving ratings. To aviod this bias, we can subtract user's average rating of all items when computing weighted average, and add it back for target user, shown as below.
```
r_ij = ave(r_i) + Sum(Similarities(u_i, u_k)*(r_kj - ave(r_k))) / num_ratings
```  

__Person Correlation__ is commonly used as the similarity measure.
- Similarity Measures
    - Pearson Correlation
    - Cosine Similarity

#### Item-based Collaborative Filtering
This methods considers neighboorhood of items, not users. __Cosine Similarity__ is commonly used as the similarity measure. We say two items are similar when they recieved similar ratings from a same user. Then, we will make prediction for a target user on an item by calculating weighted average of ratings on most X similar items from this user. One key advantage of item-based CF is the stability which is that the ratings on a given item will not change significantly overtime, unlike the tastes of human beings.  
There are quite a few limitations of this method. It __doesn't handle sparsity well__ when no one in the neighborhood rated an item that is what you are trying to predict for target user. Also, it's not computational effiecient as the growth of the number of users and products.

#### Model-based: Matrix Factorization - SVD

What matrix factorization eventually gives us is how much a user is aligbed with a set of laten features, and how much a movie fits into this set of laten features. The advantage of it over standard nearest neighborhood is that even though two users haven't rated any same movies, it's still possible to find the similarity between them if they share the similar underlying tastes, again laten features.  


Also, matrix factorizatin effectively find noises and outliers. Since newly found feature vectors are the eigen vectors of users and items, vectors that are far from the others will be outliers.




### Content-Based
use meta data such as genre, producer, actor, musician 
- requires a good amount of information of items' own features

### References
- [Introduction to Recommender System - Approaches of Collaborative Filtering: Nearest Neighborhood and Matix Factorization](https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26)  
- [Collaborative Filtering - 추천시스템의 핵심 기술](https://www.oss.kr/info_techtip/show/5419f4f9-12a1-4866-a713-6c07fd36e647)
- [How to build a Simple Recommender System in Python](https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d)  
- [Various Implementations of Collaborative Filtering](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)
- [Matrix Factorization Techniques for Recommender Systems](https://towardsdatascience.com/paper-summary-matrix-factorization-techniques-for-recommender-systems-82d1a7ace74)
- [논문][https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf]