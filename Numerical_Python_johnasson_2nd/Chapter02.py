import numpy as np 

# NUMPY ARRAYS---------------------------------------------------------
print("NUMPY ARRAYS".center(72, "-"))

'''
dtype   Variants                                Description
int     int8, int16, int32, int64               Integers
uint    uint8, uint16, uint32, uint64           Unsigned (nonnegative) integers
bool    Bool                                    Boolean (True or False)
float   float16, float32, float64, float128     Floating-point numbers
complex complex64, complex128, complex256       Complex-valued floating-point numbers
'''
data = np.array([[1,2], [3,4], [5,6]])
print(type(data))
print(data.ndim)
print(data.shape)
print(data.size)
print(data.dtype)
print(data.nbytes)


# Data types-----------------------------------------------------------
print("DATA TYPES".center(72,"-"))
array1 = np.array([1,2,3], dtype=int)
array2 = np.array([1,2,3], dtype=float)
array3 = np.array([1,2,3], dtype=complex)

print(array1)
print(array2)
print(array3)

'''
Once a numpy array is created, its dtype cannot be changed other than
creating a new copy of type-casted array values
'''

data = np.array([1,2,3], dtype=float)
print(data)
data = np.array(data, dtype=int)
print(data)
# astype method can be used here aswell
print(data.astype(float))

'''
When computing with numpy arrays, the data types might get promoted
from one type to another, if required by the operation. Ex: adding floats
to complex arrays changes the floats to complex 
'''

d1 = np.array([1,2,3], dtype=float)
d2 = np.array([1,2,3], dtype=complex)
print(d1+d2)
print((d1+d2).dtype)

'''
type casting the arrays is VERY important. Some operations will not run 
if the array type is incorrect. An example of this is operations on 
negative numbers. 
'''

d1 = np.array([-1, 0, 1])
# print(np.sqrt(d1)) runtime error because of neg. sqrt.
print(np.sqrt(d1, dtype=complex))

# Real and Imaginary parts---------------------------------------------- 
print("REAL AND IMAGINARY PARTS".center(72, "-"))
'''
Regardless of the value of the dtype attribute, all numpy arrays 
instances have the attributes real and img for extracting the real
and imaginary parts of the array 
'''

data = np.array([1,2,3], dtype=complex)
print(data)
print(data.real)
print(data.imag)

#Order of Array Data in Memory ----------------------------------------
print("ORDER OF ARRAY DATA IN MEMORY".center(72, "-"))
'''
Multidimensional arrays are stored as contiguous data in memory
However, there are multiple ways to store them. One major way is to store
the array as a consecutive sequence of rows. This is row-major format. 
Another major way is to store the columns one after another. This is
column-major format. C uses row-major format. Fortran uses col-major 
format. 
numpy arrays can be stored either way using the arguement order=
    order = 'C'
    order = 'F'
    default is row-major
this information useful when working with interefaces in other languages
nparray.stides definds exactly how the mapping the indexes is done
consider the C-order array A with shape (2,3)
    a 2D array with 2 and 3 elements along the 1st and 2nd dimensions
    if int32, total memory = 2x3x4 = 24
    strides attribute of this array (4x3, 4x1) = (12,4)
        each increment of m in A[n,m] increases storage by 4 bytes
        each increment of n in A[n,m] increases store by 12 bytes
            2nd dimenstion of array has length 3
    this array stored in 'F' would have strides equal to (4,8)

operations that only require changing the strides attribute, such as 
transpose creates new ndarray objects that reference the original data 
as the original array. These arrays are called views. 
It is important to know the distinction between which operations create 
views vs. new independant arrays
'''

# Creating Arrays -----------------------------------------------------
print("CREATING ARRAYS".center(72, "-"))

'''
np.array 
    Creates an array for which the elements are given by an array-like 
    object, which, for example, can be a (nested) Python list, a tuple, 
    an iterable sequence, or another ndarray instance.

np.zeros 
    Creates an array with the specified dimensions and data type 
    that is filled with zeros.

np.ones 
    Creates an array with the specified dimensions and 
    data type that is filled with ones.

np.diag 
    Creates a diagonal array with specified values along the 
    diagonal and zeros elsewhere.

np.arange 
    Creates an array with evenly spaced values between the specified 
    start, end, and increment values.

np.linspace 
    Creates an array with evenly spaced values between specified start 
    and end values, using a specified number of elements.

np.logspace 
    Creates an array with values that are logarithmically spaced between 
    the given start and end values.

np.meshgrid 
    Generates coordinate matrices (and higher-dimensional coordinate 
    arrays) from one-dimensional coordinate vectors.

np.fromfunction 
    Creates an array and fills it with values specified by a 
    given function, which is evaluated for each combination of indices 
    for the given array size.

np.fromfile 
    Creates an array with the data from a binary (or text) file. 
    NumPy also provides a corresponding function np.tofile with which 
    NumPy arrays can be stored to disk and later read back 
    using np.fromfile.

np.genfromtxt,np.loadtxt
    Create an array from data read from a text file, for example, 
    a commaseparated value (CSV) file. The function np.genfromtxt 
    also supports data files with missing values.

np.random.rand 
    Generates an array with random numbers that are uniformly distributed
    between 0 and 1. Other types of distributions are also available 
    in the np.random module.
'''

#Arrays Created from Lists and Other Array-Like Objects----------------
print("ARRAYS CREATED FROM LISTS AND SIMILAR THINGS".center(72,"-"))
d1 = np.array([1,2,3,4])
print(d1.ndim)
print(d1.shape)

d2 = np.array([[1,2], [3,4]])
print(d2.ndim)
print(d2.shape)

#Arrays filled with Constant Values -----------------------------------
print("ARRAYS FILLED WITH CONSTANT VALUES".center(72,"-"))

d1 = np.zeros((2,3))
print(d1)
d2 = np.ones(4)
print(d2)

# data type for np.zeros, np.ones is float64, this can be modified 
x1 = 5.4 * np.ones(10)
x2 = np.full(10, 5.4)
#np.full creates a nparray object with the given value
print(x1)
print(x2)

# an empty array can be filled with an np.fill function 
x1 = np.empty(5)
x1.fill(3.0)
print(x1)

x2 = np.full(5, 3.0)
print(x2)

# Arrays filled with incremental sequences ----------------------------
print("ARRAYS FILLED WITH INCREMENTAL SEQUENCES".center(72, "-"))
'''
np.arange(a,b,c):
    start, stop, increment
np.linspace(a,b,c):
    start, stop, number of points
'''
d1 = np.arange(0, 10, 1)
print(d1)

d2 = np.linspace(0, 10, 11)
print(d2)

# Arrays filled with logarithmic sequences ----------------------------
print("ARRAYS WITH LOG SEQUENCES".center(72, "-"))

d1 = np.logspace(0, 2, 5) # 5 data points between 10^0 and 10^2
print(d1)
# Meshgrid Arrays -----------------------------------------------------
print("MESHGRID ARRAYS".center(72, "-"))
# given 2 1D arrays, generate a 2D array with np.meshgrid 

x = np.array([-1, 0, 1])
y = np.array([-2, 0, 2])
X, Y = np.meshgrid(x,y)
print(X)
print(Y)

Z = (X+Y)**2

print(Z)
#TODO: REVIEW THIS SECTION, ASK FOR HELP IF I DON'T UNDERSTAND

#Creating uninitialized Arrays ----------------------------------------
print("CREATING UNITIALIZED ARRAYS".center(72, "-"))

x1 = np.empty(3, dtype=float)
print(x1)
'''
A good rule of thumb is to use np.zeros as opposed to np.empty since 
np.empty can create really small values which can create really hard to
find bugs 
'''

# Creating Arrays with Properties of Other Arrays ---------------------
print("CREATING ARRAYS WITH PROPERTIES OF OTHER ARRAYS".center(72,"-"))

def f(x):
    y = np.ones_like(x)
    return y 

'''
similar functions includes:
    np.ones_like
    np.zeros_like
    np.full_like
    np.empty_like
these commands are useful when you want arrays of the same 
size and dtype

'''

#Creating Matrix Arrays -----------------------------------------------
print("CREATING MATRIX ARRAYS".center(72,"-"))

d1 = np.identity(4)
print(d1)
# 1's on the diagonal, 0's everywhere else 

d2 = np.eye(3, k=1) #identity matrix offset by 1 (right one)
d3 = np.eye(3, k=-1) #identity matrix offset by -1 (left one)
print(d2) 
print(d3)
d4 = np.eye(4, k=2) #identity matrix offset by 2 (right two)
print(d4)

d5 = np.diag(np.arange(0, 30, 5))
print(d5)

# 1D array slicing and indexing ---------------------------------------
print("1D ARRAY SLICING AND INDEXING".center(72, "-"))
'''
a[m] 
    Select element at index m, 
    where m is an integer (start counting form 0).

a[-m] 
    Select the n th element from the end of the list, 
    where n is an integer. The last
    element in the list is addressed as -1, 
    the second to last element as -2, and so on.

a[m:n] 
    Select elements with index starting at m and 
    ending at n - 1 (m and n are integers).

a[:] or a[0:-1]
    Select all elements in the given axis.

a[:n] 
    Select elements starting with index 0 and 
    going up to index n - 1 (integer).

a[m:] or a[m:-1]
    Select elements starting with index m (integer) 
    and going up to the last element in the array.

a[m:n:p] 
    Select elements with index m through n (exclusive), 
    with increment p.

a[::-1] 
    Select all the elements, in reverse order.

'''

a = np.arange(0, 11)
print(a)

print(a[0]) #0
print(a[-1]) #10
print(a[4]) #4
print(a[1:-1]) #[1, ... 9]
print(a[1:-1:2]) # [1, 3, 5, 7, 9]
print(a[:5]) #[0, 1, 2, 3, 4]
print(a[-5:]) # [6, 7, 8, 9, 10]
print(a[::-2]) # [10, 8, 6, 4, 2, 0]

# Multidimensional Array slicing and indexing -------------------------
print("MULTIDIMENSIONAL ARRAY SLICIING AND INDEXING".center(72, "-"))
f = lambda m, n: n+10*m
A = np.fromfunction(f, (6,6), dtype=int)
print(A)

print(A[:,1]) #2nd col
print(A[1,:]) #2nd row
print(A[:3, :3]) # first 3 rows, first 3 cols
print(A[3:, 3:]) #last 3 row, last 3 cols
print(A[3:, :3]) # last 3 rows, first 3 cols

print(A[::2, ::2]) # every second element, starting from 0,0 
print(A[1::2, 1::3]) # every second and third element starting from 1,1

# Views ---------------------------------------------------------------
print("VIEWS".center(72,"-"))

'''
subarrays extracted from arrays using slice operations are alternative
views of the same underlying array data. 
they are the original array, just with different stride configurations
when elements in a view are assigned new values, the values in the 
original arrays are therefore also updated. 
'''
B = A[1:5, 1:5]
print(B)
B[:, :] = 0 
print(A)
# B is a view of A, when B is updated so is A

C = B[1:3, 1:3].copy()
print(C)
C[:, :] = 1 # C is a copy, not a view --> B does not change
print(C)
print(B)

#Fancy indexing and Boolean Valued Indexing 
print("FANCYING INDEXING AND BOOLEAN-VALUED INDEXING".center(72,"-"))
A = np.linspace(0, 1, 11)
print(A)
print(A[np.array([0,2,4])])
print(A[[0,2,4]]) 
# these are same thing 

'''
Boolean-valued index arrays are indexed if the element n satisfies the 
boolean condition. This is useful for filtering out elemetns 
'''
B = A > 0.5 
print(B)
print(A[A>0.5])

'''
arrays returned using fancy indexing and boolean-valued indexing are not
views but rather new independant arrays. It is possible to assign values
to elements using fancy indexing 
'''

#FANCY INDEXING

A = np.arange(10)
print(A)
indices = [2,4,6]
B = A[indices]
print(B, end=" B\n")
B[0] = 1 # this does not affect A
print(B,end=" B\n")
print(A)
A[indices] = -1 # this alters A, -1 @ indices
print(A)

# BOOLEAN-VALUED INDEXING

A = np.arange(10)
B = A[A>5]
print(B, end= "  B\n")
B[0] = -1 # does not affect A
print(A, end="  A\n")
print(B, end="  B\n")
A[A>5] = -1 # this alters A
print(A)

# COOL GRAPHIC ON PG 65 FOR SLICING 
# RESHAPING AND RESIZING ----------------------------------------------
print("RESHAPING AND RESIZING".center(72,"-"))
'''
np.reshape, np.ndarray.reshape
    Reshape an N-dimensional array. 
    The total number of elements must remain the same.

np.ndarray.flatten 
    Creates a copy of an N-dimensional array, and reinterpret it as a
    one-dimensional array (i.e., all dimensions are collapsed into one).

np.ravel, np.ndarray.ravel
    Create a view (if possible, otherwise a copy) of an 
    N-dimensional array in which it is interpreted as a one-dimensional 
    array.

np.squeeze 
    Removes axes with length 1.

np.expand_dims, np.newaxis
    Add a new axis (dimension) of length 1 to an array, where np.
    newaxis is used with array indexing.

np.transpose, np.ndarray.transpose, np.ndarray.T
    Transpose the array. The transpose operation corresponds to reversing
    (or more generally, permuting) the axes of the array.

np.hstack 
    Stacks a list of arrays horizontally (along axis 1): 
    for example, given a list of column vectors, appends the columns 
    to form a matrix.

np.vstack 
    Stacks a list of arrays vertically (along axis 0): 
    for example, given a list of row vectors, appends the rows to form 
    a matrix.

np.dstack 
    Stacks arrays depth-wise (along axis 2).

np.concatenate 
    Creates a new array by appending arrays after each other, along a
    given axis

np.resize 
    Resizes an array. Creates a new copy of the original array, with the
    requested size. If necessary, the original array will be repeated 
    to fill up the new array.

np.append 
    Appends an element to an array. Creates a new copy of the array.

np.insert 
    Inserts a new element at a given position. 
    Creates a new copy of the array.

np.delete 
    Deletes an element at a given position. 
    Creates a new copy of the array


reshaping an array does not require modifying the underlying array data
it only changes the way the data is interpreted (redefining the strides 
attribute)
'''
data = np.array([[1,2], [3,4]])
print(data)
data2 = np.reshape(data, (1,4))
print(data2)
data3 = data.reshape(4)
print(data3)

# reshaping produces a view, not a copy 

data = np.array([[1,2], [3,4]])
print(data)
print(data.flatten(), end = "   flatten\n")
print(data.ravel(), end = "     ravel\n")
print(data.flatten().shape)

'''
Introduce new axes into the array by using np.reshape or by adding 
the keyword, np.newaxis keyword at the place of the new axis
'''
data = np.arange(0,5)
print(data)
column = data[:, np.newaxis]
print(column)
row = data[np.newaxis, :]
print(row)

'''
np.expand_dims can also be used to add new dimensions 
    data[:, np.newaxis] = np.expand_dims(data, axis=1)
    data[np.newaxis, :] = np.expand_dims(data, axis=0)

np.vstack and np.hstack are very useful for combinging data frames into
higher dimensional data. 
    vstack is for vertical stacking
    hstack is for horizontal stacking 
    np.concatenate provides a similar functionality, but takes an 
    axis argument 
'''

data = np.arange(5)
print(data)
print(np.vstack((data, data, data)))
print(np.hstack((data, data, data)))

data = data[:, np.newaxis]
print(np.hstack((data, data, data)))

# Vectorized Expressions ----------------------------------------------
print("VECTORIZED EXPRESSIONS".center(72, "-"))

'''
a binary operation involving two arrays is well defined if the arrays can 
be broadcasted into the same shape and size. 

examples:
    between scalar and the array 
    the scalar between distributed and the operation applied to each 
    element in the array

When the expression contains arrays of unequal sizes, the operations can
still be well defined if the smaller of the arrays can be broadcasted
(effictively expanded) to match the larger array 


'''

# Arithmetic Operations -----------------------------------------------
print("ARITHMETIC OPERATIONS".center(72, "-"))

x = np.array([[1,2], [3,4]])
y = np.array([[5,6] , [7,8]])
print(x+y)
print(y-x)
print(x-y)
print(x*y)
print(y/x)
print(x*2)
print(2**x)
print(y/2)
print((y/2).dtype)

x = np.array([1,2,3,4]).reshape(2,2)
z = np.array([1,2,3,4])
#print(x/z)

z = np.array([[2,4]])
print(z.shape)
# now when dividing x by z, it is the equivalent of dividing x by zz
print(x/z)
zz = np.concatenate([z,z], axis= 0)
print(zz)
print(x/zz)
# divides [1,2,3,4] by [2,4,2,4]
z = np.array([[2], [4]])
print(z.shape)
print(x/z)
#divides [1,2,3,4] by [2,2,4,4] 
zz = np.concatenate([z,z], axis=1)
print(zz)
print(x/zz)

'''
Operation Table

+, += Addition
-, -= Subtraction
*, *= Multiplication
/, /= Division
//, //= Integer division
**, **= Exponentiation

'''

# Elementwise Functions -----------------------------------------------
print("ELEMENTWISE FUNCTIONS".center(72,"-"))
'''
Table of Elementwise Elementary mathematical functions

np.cos, np.sin, np.tan                  Trigonometric functions.
np.arccos, np.arcsin, np.arctan         Inverse trigonometric functions.
np.cosh, np.sinh, np.tanh               Hyperbolic trigonometric functions.
np.arccosh, np.arcsinh, np.arctanh      Inverse hyperbolic trigonometric functions.
np.sqrt                                 Square root.
np.exp                                  Exponential.
np.log, np.log2, np.log10               Logarithms of base e, 2, and 10, respectively

'''

x = np.linspace(-1, 1, 11)
print(x)
y = np.sin(np.pi*x)
print(np.round(y, decimals=4))
d1 = np.add(np.sin(x)**2, np.cos(x)**2)
print(d1)
print(np.sin(x)**2 + np.cos(x)**2)

'''
Table of Numpy Functions for Elementwise Mathematical Operations
np.add              Addition
np.subtract         Subtraction
np.multiply         Multiplication
np.divide           Division
np.power            Raises first input argument to the power of the 
                    second input argument (applied elementwise).
np.remainder        The remainder of division.
np.reciprocal       The reciprocal (inverse) of each element.

np.real, np.imag, np.conj
                
        The real part, imaginary part, and the complex conjugate of the
        elements in the input arrays.

np.sign, np.abs                     The sign and the absolute value.
np.floor, np.ceil, np.rint          Convert to integer values.
np.round                            Rounds to a given number of decimals.

np.vectorize function takes a nonvectorized function and returns a 
vectorized function 
'''

def heaviside(x):
    return 1 if x>0 else 0
print(heaviside(-1))
print(heaviside(1.5))
x = np.linspace(-5, 5, 11)
print(x)
heaviside = np.vectorize(heaviside)
print(heaviside(x))

# Aggregate functions -------------------------------------------------
print("AGGREGATE FUNCTIONS".center(72, "-"))
'''
takes an array as input and by default returns a scalar
ex: avgs, standard deviations, variances, etc... 
'''
data = np.random.normal(size=(15,15))
#print(data)
print(np.mean(data))
print(data.mean())
'''
Table of common aggregate functions 

np.mean         The average of all values in the array.
np.std          Standard deviation.
np.var          Variance.
np.sum          Sum of all elements.
np.prod         Product of all elements.
np.cumsum       Cumulative sum of all elements.
np.cumprod      Cumulative product of all elements.
np.min, np.max  
                The minimum/maximum value in an array.

np.argmin, np.argmax 
                The index of the minimum/maximum value in an array.

np.all 
        Returns True if all elements in the argument array are nonzero.
np.any 
        Returns True if any of the elements in the argument array is nonzero

'''

data = np.random.normal(size=(5,10,15))
#print(data)
print(data.sum(axis=0).shape)
print(data.sum(axis=(0,2)).shape)
print(data.sum())
data = np.arange(1,10).reshape(3,3)
print(data)
print(data.sum()) #all of it
print(data.sum(axis=0)) # sum by column
print(data.sum(axis=1)) # sum by row

# Boolean Arrays and Conditional Expressions --------------------------
print("BOOLEAN ARRAYS AND CONDITIONAL EXPRESSIONS".center(72, "-"))
'''
Numpy Arrays can use all the standard comparison operators
    <, >, <=, >=, !=, ==
'''
a = np.array([1,2,3,4])
b = np.array([4,3,2,1])
print(a<b)

'''
This is a common use case of np.all or np.any aggregate functions to 
quickly determine values in the arrays 

np.all returns true if ALL elements in the array are nonzero
np.any returns true if any elements in the array are nonzero
'''
print(np.all(a<b))
print(np.any(a<b))

if np.all(a<b):
    print("all elements in a are smaller than same index element in b")
elif np.any(a<b):
    print("some elements in a are smaller than same index element in b")
else:
    print("all elements in b are smaller than same index element in a")

'''
Boolean-valued arrays are useful because they avoid conditionals if 
statements all together and they keep computation in vectorized form
'''
x = np.array([-2, -1, 0, 1, 2, 3, 4])
print(x>0)
print(1*(x>0))
print(x*(x>0))

def pulse(x, position, height, width):
    return height * (x>= position) * (x<= (position+width))
x = np.linspace(-5, 5, 11)
print(pulse(x, position=-2, height=1, width=5))
# i did not know arguments could be spelled out like this
#print(pulse(x, -2, 1, 5))
print(pulse(x, position=1, height=1, width=5))
'''
(x>= position) * (x <= (position + width)) is a multiplication of two
boolean values arrays. In this instance the * operator acts as an 
element wise AND operator. 

This function can also be rewritten the following way:
'''

def pulse(x, position, height, width):
    return height * np.logical_and(x>= position, x <= (position + width))

x = np.linspace(-4, 4, 9)
print(np.where(x<0, x**2, x**3))
# squares false values, cubes true values 

'''
Table of Numpy Functions for conditional and logical expressions

np.where        Chooses values from two arrays depending on the value
                of a condition array.
np.choose       Chooses values from a list of arrays depending on the
                values of a given index array.
np.select       Chooses values from a list of arrays depending on a list
                of conditions.
np.nonzero      Returns an array with indices of nonzero elements.
np.logical_and  Performs an elementwise AND operation.
np.logical_or   Elementwise OR operations 
np.logical_xor  Elementwise XOR operations.
np.logical_not  Elementwise NOT operation (inverting).
'''
print(x)
print(np.select([x<-1, x<2, x>=2], [x**2, x**3, x**4]))
'''
if the element in x is i<-1, i**2
if the element in x is i<2, i**3
if the element in x is i>=2, i**4
'''
print(np.choose([0,0,0,1,1,1,2,2,2], [x**2, x**3, x**4]))
# applies the index of the second list to the corresponding index in x
print(x)
print(np.nonzero(abs(x)>2)) #returns indices where condition is True
print(x[np.nonzero(abs(x)>2)]) #returns sliced array
print(x[abs(x)>2])

# Set Operations ------------------------------------------------------
print("SET OPERATIONS".center(72, "-"))
'''
Numpy functions for operating on sets

np.unique 
    Creates a new array with unique elements, where each value 
    only appears once.
np.in1d 
    Tests for the existence of an array of elements in another array.
np.intersect1d 
    Returns an array with elements that are contained in two given arrays.
np.setdiff1d 
    Returns an array with elements that are contained in one, 
    but not the other, of two given arrays.
np.union1d 
    Returns an array with elements that are contained in either,
     or both, of two given arrays.
'''

a = np.unique([1,2,3,3])
b = np.unique([2,3,4,4,5,6,5])
print(a)
print(b)
print(np.in1d(a,b)) # elements of a in array b 
print(np.in1d(b,a)) # element of b in array a

print(1 in a)
print(1 in b)

#testing subsets 
print(np.all(np.in1d(a,b)))
# returns true is np.in1d(a,b) == [True, True, .... True]

print(np.union1d(a,b))
print(np.intersect1d(a,b))
print(np.setdiff1d(a,b)) # elements in a but not in b
print(np.setdiff1d(b,a)) # elements in b but not in a 
#Operations on Arrays -------------------------------------------------
print("OPERATIONS ON ARRAYS".center(72, "-"))
data = np.arange(9).reshape(3,3)
print(data)
print(np.transpose(data))
# ndarray.T also works; reverses all axes 
#print(np.random.rand(1,2,3,4,5))
data = np.random.randn(1,2,3,4,5)
print(data.shape)
print(data.T.shape)

'''
Summary of Numpy Functions for Array Operations 

np.transpose, np.ndarray.transpose, np.ndarray.T
        The transpose (reverse axes) of an array.

np.fliplr/np.flipud 
        Reverse the elements in each row/column.

np.rot90 
        Rotates the elements along the first two axes by 90 degrees.

np.sort, np.ndarray.sort
        Sort the elements of an array along a given specified axis 
        (which default to the last axis of the array). 
        The np.ndarray method sort performs the sorting in place, 
        modifying the input array
'''

#Matrix and Vector Operations -----------------------------------------
print("MATRIX AND VECTOR OPERATIONS".center(72, "-"))
'''
Summary of Numpy Functions for Matrix Operations 

np.dot 
    Matrix multiplication (dot product) between two given arrays 
    representing vectors, arrays, or tensors.

np.inner 
    Scalar multiplication (inner product) between two arrays 
    representing vectors.

np.cross 
    The cross product between two arrays that represent vectors.

np.tensordot 
    Dot product along specified axes of multidimensional arrays.

np.outer 
    Outer product (tensor product of vectors) between two arrays 
    representingvectors.

np.kron 
    Kronecker product (tensor product of matrices) 
    between arrays representing matrices and higher-dimensional arrays.

np.einsum 
    Evaluates Einstein's summation convention for multidimensional arrays

'''

A = np.arange(1, 7).reshape(2,3)
B = np.arange(1,7).reshape(3,2)
print(A)
print(B)
print(np.dot(A,B))
print(np.dot(B,A))

C = np.arange(9).reshape(3,3)
D = np.arange(3)
print(C)
print(D)
print(np.dot(C, D))
print(C.dot(D))
A = np.random.rand(3,3)
B = np.random.rand(3,3)
print(A)
print(B)

#A' = BAB^-1
Ap = np.dot(B, np.dot(A, np.linalg.inv(B)))
# or 
Ap = B.dot(A.dot(np.linalg.inv(B)))
# or 
Ap = B @ A @ np.linalg.inv(B)

print(Ap)
# DON'T DO THAT BECAUSE THERE IS A BETTER WAY 

A = np.matrix(A)
B = np.matrix(B)
Ap= B * A * B.I
print(Ap)
'''
matrix class is often discouraged because it is not clear if the 
* operator is signals element wise or matrix multiplication
A good use case would be to explictly cast matrixes uses np.asmatrix, 
and converting the output back to np.asarray
'''
A = np.asmatrix(A)
B = np.asmatrix(B)
Ap = B * A * B.I
Ap = np.asarray(Ap)

x = np.arange(3)
print(x)
print(np.inner(x,x))
print(np.dot(x,x))

''' 
the main difference is that np.inner expects two input arguemtns with the
same dimensions while np.dot can take input vectors of shape 1xN and Nx1
'''
y = x[:, np.newaxis]
print(y)
print(np.dot(y.T, y))

'''
While the inner product maps two vectors to a scalar, the outer product
performs the complementary opertion of mapping two vectors to a matrix

'''
x = np.array([1,2,3])
print(np.outer(x,x))

'''
The outer product can also be calculated using the Kronecker product 
using the function np.kron which produces and an output array with shape
(M*P, N*Q) if the input arrays have shapes (M, N) and (P,Q) respectively
Kronecker product for 2 1D arrays is (M*P)

'''
print(np.kron(x,x))
print(np.kron(x[:, np.newaxis],x[np.newaxis, :]))

'''
np.outer function is primarily intended for vectors as input, the np.kron
function can be used for computing tensor products or arrays of arbitrary 
dimensions (both inputs must have the same number of axes)

Example: tensor product of two 2x2 matrices
'''

print(np.kron(np.ones((2,2)), np.identity(2)))
print(np.kron(np.identity(2), np.ones((2,2))))

'''
Einstein's summation convention 
    implicit summation is assumed over each index that occurs multiple 
    times in an expression.
Examples: 
    the scalar produce between two vectors x and y is expressed as XnYn
    the matrix multiplication of two matrices A and B is expressed as 
    AmkBkn.
First argument is an index epxrresion
'''
x = np.array([1,2,3,4])
y = np.array([5,6,7,8])

print(np.einsum("n,n", x, y))
print(np.inner(x,y))

A = np.arange(9).reshape(3,3)
B = A.T
np.einsum("mk,kn", A, B)