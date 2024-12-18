# eigenestimation


```
source env/bin/activate
pip3 install -r requirements.txt
python -m ipykernel install --user --name=env --display-name "Python (eigenestimation)"
```

Below is a more formal proof-style argument suitable for inclusion in a technical paper.

---

**Proposition:**  
Let $P(y \mid x, w)$ be a parametric model over a finite label space \(\{1,\ldots,K\}\), and let \( w' \) be a fixed parameter vector. Define the following divergences:

1. \(\mathcal{L}(w) := \text{KL}(P(\cdot \mid x,w) \;||\; Q(\cdot \mid x))\), where \( Q(\cdot \mid x) \) is independent of \( w \).
2. \(\mathcal{D}(w) := \text{KL}(P(\cdot \mid x,w) \;||\; P(\cdot \mid x,w'))\).

Suppose there exists a set of vectors \(\{v_i\}_{i=1}^m \subset \mathbb{R}^d\) (where \( d = \dim(w) \)) whose linear span can be used to represent \(\nabla_w \mathcal{L}(w')\). We will show that the same set \(\{v_i\}\) is sufficient to represent \(\nabla_w^2 \mathcal{D}(w')\).

**Proof:**

1. **Representation of the First-Order Derivative:**

   Consider \(\mathcal{L}(w) = \text{KL}(P(\cdot \mid x,w) \| Q(\cdot \mid x))\). By definition:
   \[
   \mathcal{L}(w) = \sum_{y=1}^K P(y \mid x,w) \log \frac{P(y \mid x,w)}{Q(y \mid x)}.
   \]
   
   Differentiating w.r.t. \( w \) at \( w' \):
   \[
   \nabla_w \mathcal{L}(w') = \sum_{y=1}^K A_y(w') \nabla_w \log P(y \mid x,w'),
   \]
   for some coefficients \( A_y(w') \) that depend on \( P(\cdot \mid x,w') \) and \( Q(\cdot \mid x) \).

   Define:
   \[
   G_y(w') := \nabla_w \log P(y \mid x,w').
   \]
   Thus:
   \[
   \nabla_w \mathcal{L}(w') = \sum_{y=1}^K A_y(w') G_y(w').
   \]

   By assumption, \(\nabla_w \mathcal{L}(w')\) can be represented as a linear combination of the vectors \(\{v_i\}_{i=1}^m\). Therefore, there exist scalars \(\{\alpha_i\}\) such that:
   \[
   \nabla_w \mathcal{L}(w') = \sum_{i=1}^m \alpha_i v_i.
   \]

   From this it follows that each \( G_y(w') \) (and hence any linear combination of them) lies in the linear span of \(\{v_i\}\). In other words:
   \[
   \text{span}\{G_y(w') : y=1,\dots,K\} \subseteq \text{span}\{v_i : i=1,\dots,m\}.
   \]

2. **Representation of the Second-Order Derivative:**

   Now consider \(\mathcal{D}(w) = \text{KL}(P(\cdot \mid x,w) \| P(\cdot \mid x,w'))\). By definition:
   \[
   \mathcal{D}(w) = \sum_{y=1}^K P(y \mid x,w) \log \frac{P(y \mid x,w)}{P(y \mid x,w')}.
   \]
   
   At \( w = w' \), \(\mathcal{D}(w') = 0\). The Hessian at \( w' \) is given by the Fisher Information matrix:
   \[
   \nabla_w^2 \mathcal{D}(w') = I(w'),
   \]
   where
   \[
   I(w') = \sum_{y=1}^K P(y \mid x,w') G_y(w') G_y(w')^\top.
   \]

   Thus, the Hessian \(\nabla_w^2 \mathcal{D}(w')\) is a weighted sum of outer products \(G_y(w') G_y(w')^\top\).

3. **Expressing the Hessian Using the Same Set of Vectors:**

   Since \(G_y(w') \in \text{span}\{v_i\}\) for each \(y\), we can write:
   \[
   G_y(w') = \sum_{i=1}^m \beta_{y,i} v_i,
   \]
   for some scalars \(\beta_{y,i}\).

   Substitute this into the expression for \( I(w') \):
   \[
   I(w') = \sum_{y=1}^K P(y \mid x,w') \left(\sum_{i=1}^m \beta_{y,i} v_i\right) \left(\sum_{j=1}^m \beta_{y,j} v_j\right)^\top.
   \]

   Expanding the outer product:
   \[
   I(w') = \sum_{y=1}^K P(y \mid x,w') \sum_{i=1}^m \sum_{j=1}^m \beta_{y,i}\beta_{y,j} v_i v_j^\top.
   \]

   This is a finite linear combination of matrices of the form \(v_i v_j^\top\). Each \(v_i v_j^\top\) is constructed solely from the vectors \(\{v_i\}\).

   Hence \( I(w') \) (and thus \(\nabla_w^2 \mathcal{D}(w')\)) lies in the space spanned by the outer products of the vectors \(\{v_i\}\). Since each \(v_i\) was part of the original set assumed to approximate \(\nabla_w \mathcal{L}(w')\), it follows that the same set \(\{v_i\}\) is sufficient to represent the Hessian \(\nabla_w^2 \mathcal{D}(w')\).

**Conclusion:**
We have shown that if a set of vectors \(\{v_i\}\) can be used to represent the gradient \(\nabla_w \mathcal{L}(w')\), then the same set of vectors can be used to represent the Hessian \(\nabla_w^2 \mathcal{D}(w')\). This holds because both first-order and second-order derivatives at \( w' \) are ultimately composed of the same building blocks, \(\{G_y(w')\}\), and any vector space containing these gradients also contains all outer products needed to form the Hessian.