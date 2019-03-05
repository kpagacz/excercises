
##Introduction
This is a demo of non-negative matrix factorization. It showcases how the NMF works on a random dataset and aims to enhance the understanding of the method.

First imports:


```python
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
```


```python
orig = np.random.randint(low = 0, high = 10, size = 15).reshape(3, 5)
orig = pd.DataFrame(orig, index = ['A', 'B', 'C'], columns = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
print(orig)
```

       Feature 1  Feature 2  Feature 3  Feature 4  Feature 5
    A          8          2          6          7          8
    B          3          0          2          5          0
    C          0          6          4          1          1
    

Exemplary dataset contains 3 samples. Each sample has 5 distinct features.


```python
nmf = NMF(n_components = 2)
transformed = nmf.fit_transform(orig)
print('Transformed are:\n {}'.format(transformed))
components = nmf.components_
print('Components are:\n{}'.format(features))
```

    Transformed are:
     [[3.60017113e+00 9.75377670e-01]
     [1.37488272e+00 0.00000000e+00]
     [2.91855955e-03 2.82341819e+00]]
    Components are:
    [[2.21700983 0.         1.30721035 2.09992816 1.82815341]
     [0.         2.11713041 1.40589378 0.25325465 0.46976818]]
    

###Components
The components contain information about the original features. Each component represents some features. Let us annotate the component and look at them.


```python
comp_df = pd.DataFrame(components, columns = orig.columns, index = ['Component 1', 'Component 2'])
print(comp_df)
```

                 Feature 1  Feature 2  Feature 3  Feature 4  Feature 5
    Component 1    2.21701    0.00000   1.307210   2.099928   1.828153
    Component 2    0.00000    2.11713   1.405894   0.253255   0.469768
    

First component mainly tells about feature 1 and feature 4. Second component highlights feature 2 and feature 3.

###Transformed samples
Each row of transformed matrix relates to one samples in the original data and columns determin the amount by which the sample is influenced by each component. Let us annotate them and looke at them.


```python
transf_df = pd.DataFrame(transformed)
```

The idea is Transformed X Components = original data.

Let us check that fact. First we calculate the dot product of the two matrices. 


```python
prod = np.dot(transformed, components)
print('The dot product is: \n {}'.format(prod))
print('The shape of the product is: {}'.format(prod.shape))
```

    The dot product is: 
     [[7.98161478e+00 2.06500173e+00 6.07745838e+00 7.80711966e+00
      7.03986651e+00]
     [3.04812849e+00 0.00000000e+00 1.79726092e+00 2.88715493e+00
      2.51349652e+00]
     [6.47047520e-03 5.97754451e+00 3.97324124e+00 7.21172549e-01
      1.33168759e+00]]
    The shape of the product is: (3, 5)
    

Size of the product is the same as the original data. Let us look, whether the values are similar compared to the original. We calculate the difference.


```python
print('The matrix of differences is:\n{}'.format(orig.values - prod))
```

    The matrix of differences is:
    [[ 0.01838522 -0.06500173 -0.07745838 -0.80711966  0.96013349]
     [-0.04812849  0.          0.20273908  2.11284507 -2.51349652]
     [-0.00647048  0.02245549  0.02675876  0.27882745 -0.33168759]]
    

Seems that NMF wasn't far off. NMF is cute to interpret. I wonder how to visualize it.

TO-DO:
- [ ] Proof-read
- [ ] Add visualization
- [ ] Publish somewhere
