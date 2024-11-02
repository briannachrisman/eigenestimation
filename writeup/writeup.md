# Feature and Circuit Identification through Sparse Eigendecomposition

## Some terms

### Model
The output of model $f$ (of a given architecture), depends on two inputs - a set of parameters $W$ and a set of feature $x$.

After training a given model, we end up with a set of weights/parameters $W_0$.


### Divergence
The divergence metric (not called loss, because I use that term later) should describe how much a model's output changes (for a given sample) when it's parameters are altered. 


In the case of a transformer, we can compute the normalized KL-divergence. The KL-divergence describes the divergence between probability distributions and we can normalize this metric to have a minimum of zero using the following equation (note that the unnormalized KL divergence does not necessarily have a minimum at 0). We could potentially use a normalized cross entropy here as well. 

$
D(f, x, W) = KL(f(W, x), f(W_0, x)) - KL(f(W_0, x), f(W_0, x))
$

In the case of a regression problem, we can use the mean squared error loss as our divergence metric. We don't need to normalize it because MSE does have a guarenteed minimum at 0. 

$
D(f, x, W) = MSE(f(W, x), f(W_0, x))
$

### Sample-level Hessian 

We define the sample-level hessian as the second derivative of a model's loss with respect a set of parameters $w$, evaluated at $W0$ and X. 

$
H(x,w) = \nabla^{2}_{w} D(f, x, W)
$

Using the Hessian, we can compute the second derivative of the loss with respec to any direction in weight/parameter space (given by vector $u$)  by:

$
 \nabla^{2}_{u} D(f, x, W) = u(u H(x,w))^T
$

### Learning feature vectors

Our goal is to identify important directions (${u}$ or $U$) in parameter space of this Hessian where:
(1) For some samples, $\nabla^{2}_{u} D$ is very high. Moving parameters in the direction of $u$ dramatically changes the output of the model. 
(2) There is low sample-level interference between each $u$ vector. $u$ vectors should either be nearly orthogonal, or if they have high cosine similarities, they should not both have high $\nabla^{2}_{u} D$ for a given sample. 

We wish to learn a set of u-vectors ($U$) that satisfy these conditions. We will therefore minimize two losses. 

To satisfy the first goal we minimize:

$
L_{\text{low interference}}(x) = \sum_{u}(\nabla^{2}_{u} D(f, x, W_0))^2
$

To satisfy the second goal, we also want to minimize:

$
L_{\text{low interference}}(x) = \|\nabla^{2}_{u1}D \space u_1 (\nabla^{2}_{u2}D \space u_2)^{T} \|_{u_1, u_2}
$


### Optimization tricks 
Optimizing losses that depend on a full hessian is computationally expensive. We use two tricks to minimize these losses.

#### Jacobian-vector products
First, Instead of forming the full hessian, or full jacobian, we can compute nested jacobian-vector products.

$
\nabla^{2}_{u} D(f, X, W_0) = \nabla_{w} (\nabla_{w} D(f, X, W_0) \space u ) \space u
$


#### Bi-level optimization

We use two separate optimization loops and perform bi-level optimization. 

    For x_batch in X:
        L1 = 0
        d2D_all = []

        # First optimization step.

        For u_batch in U:
        
            # Compute steep hessian loss.
            1. d2D(U) = second derivative of D(f, x, W_0) with respect to u_batch.
            2. L1 = L1 + sum(d2D^2) # Compute first loss. 
            3. d2D_all = d2D_all.append(d2D(U))
        
        
        U = U + L1*step    # Update U

        # Second optimization step. 
        L2 = || (U d2D_all) (U d2D_all)^T  || # Compute low-interference loss. 
        U = U + L2*step # Update U.


Note that in our first optimization loop (minimizing $L_{\text{low itnerference}}$), we can compute L in batches of U.


## Toy models

### XOR Model

### Toy model's of superposition 

### GPT2

## Results







