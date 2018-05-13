# TENSOR FLOW - INTRODUCTION

## Graphs

![Graph](assets/markdown-img-paste-20180512143452758.png)

Every element in tensor flow is a node of a graph: x, y and a. _a_ is an add opertor.

To visualize the value you need to create a session and evaluate the graph underlaying a.
A Session object encapsulates the environment in which Operation objects are
executed and Tensor objects are evaluated.
``` Python
import tensorflow as tf
a = tf.add(3, 5)
print a
>> Tensor("Add:0", shape=(), dtype=int32)
sess = tf.Session()
    print sess.run(a)
    sess.close() # will print 8

or

with tf.Session() as sess:
    print sess.run(a)
```


### Subgraphs

![Subgraphs](assets/markdown-img-paste-20180512144819701.png)

In a more complex graph, tf engine evaluates only the necessary subgraphs. I
n this case, the useless Operation
won't be evaluated when calling the evaluation of Pow.

``` Python
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.mul(x, y)
useless = tf.mul(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
    z = sess.run(pow_op)
    tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None) # pass all variables whose values you want to a list in fetches
```

#### How to visualize a graph in tensorboard
``` Python
import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs, sess.graph) # add this line to use TensorBoard.
    print sess.run(x)
    writer.close() # close the writer when you’re done using it
```

and then from the console

``` batch
python [yourprogram].py
tensorboard --logdir="./graphs" --port 6006
```

## Operators

tf.constant and tf.variable:
 - Constant values can't be changed and are stored in the graph definition
 - Constants are initialized with a value
 - Variables can be changes in the future
 - Variables *have to be* initialized with values or an operator (REMEMBER!)
 - Sessions allocate memory to store variable values
 - tf.global_variables_initializer()

tf.placeholder and feed_dict:
 - Feed values into placeholders by dictionary (feed_dict)
 - You can feed values in variables too

##### Examples in a regression model:
 - X, Y are placeholders
 - W, b are Variables, which will be trained

### Constants

```tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)```

Defined similarly to numpy
``` Python
import tensorflow as tf
a = tf.constant([2, 2], name="a")
b = tf.constant([[0, 1], [2, 3]], name="b")
x = tf.add(a, b, name="add")
y = tf.mul(a, b, name="mul")
with tf.Session() as sess:
    x, y = sess.run([x, y])
    print x, y
    # prints >>  [5 8] [6 12]
```

Other examples of constants:
``` Python
tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]
tf.ones(shape, dtype=tf.float32, name=None)

# input_tensor is [0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
```

#### Ranges
Mind you  **tensor objects are not iterable indeed `for _ in tf.range(4): # TypeError`**
``` Python
tf.linspace(start, stop, num, name=None) # slightly different from np.linspace
tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 13.0]
tf.range(start, limit=None, delta=1, dtype=None, name='range')
# 'start' is 3, 'limit' is 18, 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
# 'limit' is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]
```

#### Randomly Generated Constants
`tf.set_random_seed(seed)`

``` Python
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
```

#### Operations
``` Python
a = tf.constant([3, 6])
b = tf.constant([2, 2])
tf.add(a, b) # >> [5 8]
tf.add_n([a, b, b]) # >> [7 10]. Equivalent to a + b + b
tf.mul(a, b) # >> [6 12] because mul is element wise
tf.matmul(a, b) # >> ValueError
tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1])) # >> [[18]]
tf.div(a, b) # >> [1 3]
tf.mod(a, b) # >> [1 0]
```

#### Numpy integration
TensorFlow integrates *seamlessly* with NumPy
Can pass numpy types to TensorFlow ops

``` Python
tf.int32 == np.int32 # True
tf.ones([2, 2], np.float32) # ⇒ [[1.0 1.0], [1.0 1.0]]
For tf.Session.run(fetches):
    # If the requested fetch is a Tensor , then the output of will be a NumPy ndarray
# TODO : REVIEW THIS
```

### Variables
Variables are a class with proper methods, constants are Operators.

``` Python
# create variable a with scalar value
a = tf.Variable(2, name="scalar")

# create variable b as a vector
b = tf.Variable([2, 3], name="vector")

# create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")

# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]))

```

Some mothods come with Variables:
``` Python
x = tf.Variable(...)
x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more
```

And they need to be initialized
``` Python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

To evaluate a variable:
``` Python
# W is a random 700 x 100 variable object
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print W.eval()
```

###### Assignement
`Variable.assign() - W.assign(100)` doesn’t assign the value 100
to W. It creates an assign op, and that op needs to be run to take effect.
You don’t need to initialize variables because assign_op does it for you.
Look at the following two examples
``` Python
# Example 1
tf.Variable.assign()
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print W.eval() # >> 10
# -------- #
# Example 2
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print W.eval() # >> 100
```

Another example:
``` Python
# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var")
# assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var_times_two) # >> 4
    sess.run(my_var_times_two) # >> 8
    sess.run(my_var_times_two) # >> 16
# It assign 2 * my_var to my_var every time my_var_times_two is fetched.
# Therefore the value of the variable is being updated!
```

### Placeholders
Generally a TF program often has 2 phases:
1. Assemble a graph
2. Use a session to execute operations in the graph.

⇒ TF can assemble the graph first without knowing the values needed for
computation. Analogy: can define the function $f(x, y) = x*2 + y$ without knowing value of x or y. *x, y are
placeholders* for the actual values.
We, or our clients, can later supply their own data when they
need to execute the computation.
``` Python
tf.placeholder(dtype, shape=None, name=None)
# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
with tf.Session() as sess:
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
    # fetch value of c
    print sess.run(c, {a: [1, 2, 3]}) # the tensor a is the key, not the string ‘a’
```

## Lazy loading

Differences between normal and lazy evaluation
``` Python
# Normal loading:
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) # >>>>> you create the node for add node before executing the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
      sess.run(z) # >>> here we pass a predefined operator
      writer.close()
```

Output of the graph is :
``` Json
node {
 name: "Add"
 op: "Add"
 input: "x/read"
 input: "y/read"
 attr {
 key: "T"
 value {
 type: DT_INT32
 }
 }
}
```

``` Python
# Lazy loading:
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
for _ in range(10):
    sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
    writer.close()
```

Output of the graph is:
```
node {
 name: "Add"
 op: "Add"
 ...
 }
...
node {
 name: "Add_9"
 op: "Add"
 ...
}
```

**In the second case we created the operator thousands of times.**

therefore remember:

1. Separate definition of ops from computing/running ops
2. Use Python property to ensure function is also loaded once the first time it is
called
