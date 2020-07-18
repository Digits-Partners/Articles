## PCA steps reminder
mean normalize the dataset
derive the covar matrix
perform SVD (singular value decomposition) whitch results in 3 matrices 
  1. eigeinvectors stacked column wise (first matrice) - gives the direction of uncorrelated features
  2. eigenvalues on the diagonal (second matrice) - the percentage of variance retained by each eigenvector
  3. 


## Transform vector steps reminder WE transfom
Using graident descent (GD), solve for R a minimization the loss function L
L = Square of Frobenius norm of the difference between the WE-target and innner product WE-orig.R
GD reminder:
compute L gradient with respect to R
update R:= R - alpha.grad_L


## K nearest neighbors steps reminder
the WE transformation does not result in an exact match of target WE, hence, given the transformed WE, one need to lookup for the closest target WE. this can be done through K-means algorithm


## Hash tables steps reminder

## Divide vector space into region steps reminder

## Locality sensitive hashing steps reminder

## Approximated nearest neighbors steps reminder

