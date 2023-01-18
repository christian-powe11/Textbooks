from turtle import * 

'''
1.2 Creating Objects 
    type = class
    built in types 
        list
        int 
        float
        string
        dictionary
1.2.1 Literal Values
    used when we want to set some variable to a specific value within a
    program
    6 in x=6
1.2.2 Non-literal object creation 
    variable = type(other_object_values)
    cat = 7
    dog = string(cat)
'''
# Calling Methods on objects ------------------------------------------
print("Calling methods on objects".center(72, "-"))

x = 'how are you'
y = x.upper()
print(x)
print(y)

'''
Mutator methods:
    actually change the existing object
    mutator methods cannot be undone 
Accessor methods:
    access methods access the current state of an object but do not 
    change the object 
    accessor methods return new object reference when called
'''

x = [1, 2, 3]
x.reverse()
print(x)
'''
some classes have mutator methods and some do not
For example, the list class has mutator methods like reverse()
string class does not have any mutator methods 
when a class contains no mutator methods, it is immutable
int and float are also immutable

1.4 implementing a class 
objects contain data and methods operate on that data. 
a class is the definition of the data and methods for a specific type 
of object 

1.4.1 creating objects and calling methods


1.5 Operator Overloading 
    the methods that begin and end with two underscores are methods that
    python associates with a corresponding operator 
    x + y ----> x.__add__(y)
class operator_overloading_cheat_sheet:
    def __add__(self,y):
        return 
        # x+y
    # the addition of two objects. the type of x determines which add operator is called 
    def __contains__(self,y):
        return
        # y in x 
    # when x is a collection you can test to see if y is in it
    def __eq__(self,y):
        return 
        # x==y
    # returns True or False depedning on the values of x and y 
    def __ge__(self,y):
        return 
        # x>= y 
    # returns True or False depedning on the values of x and y
    def __getitem__(self,y):
        return 
        # x[y]
    # returns the item at the yth position in x 
    def __gt__(self,y):
        return
        # x>y
    # returns true or false depending on the values of x and y 
    def __hash__(self):
        return
        # hash(x)
    # returns an integral value for x 
    def __int__(self):
        return 
        # int(x)
    # returns an integer represention of x 
    def __iter__(self):
        return
        # for v in x
    # returns an iterator object for the sequence x 
    def __le__(self,y):
        return 
        # x <= y 
    # returns true or false depening on the values of x and y 
    def __len__(self):
        return
        #  len(x)
    # return returns the size of x where x has some length attribute 
    def __lt__(self,y):
       pass 
        # x < y
    # returns True or False depending on the values of x and y 
    def __mod___(self,y):
        pass 
        # x % y 
    # returns the value of x modulo y
    def __mul__(self, y): 
        pass 
        # x * y
    # return the value of x*y
    def __ne__(self,y):
        pass
        # x != y 
    # returns True or false depening on the values of x and y
    def __neg__(self):
        pass 
        # -x 
    #  return the unary negation of x 
    def __repr__(self):
        pass
        # repr(x)
    # returns a string version of x suitable to evaluated by the val function 
    def __setitem__(self, i, y):
        pass
        # x[i]=y 
    # sets the item at the ith position in x to y 
    def __str__(self):
        pass
        # str(x)
    # return a string representation of x suitable for user-level interaction 
    def __sub__(self, y):
        pass
        # x - y 
    # The difference of two objects.

1.8 

writing the if statement to call the main function is only true if being
run as a stand alone program. it will not work during an import

1.9 reading from a file 


'''
