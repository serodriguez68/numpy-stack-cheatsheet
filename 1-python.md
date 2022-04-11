# Running Modes
* Interactive shell: `python`
* Running python scripts: `python myscript.py`
```python
# myscript.py script will need to have this if you want it to do anything when calling it directly
# This means: when this file is called as the main entrypoint of a program, run the code inside the if statement
if __name__ == '__main__':
    doSomething()
```

# Getting help
* `dir(s)`: lists all methods that object `s` responds to.
* `help(s.find)`:  prints help for the method `find`.

# Strings
* Single quotes or double quotes are ok
* Concatenate with `+`
  * String interpolation with f strings `f'{lower_first}.{lower_last}@example.com'`
* `len` to get length of string
* Splitting a string: `string.split(separator)`. Space is the default separator.

# Data Structures
## Lists 
* `[1, "a", [2,3]]`
* Can be of any type
* Ordered (order matters)
* `mylist.pop()` removes the last element and returns it
* `mylist.append("something")`
* `mylist[1:4]`: returns elements from indexes 1 to 3 (start to end -1)
* `len(mylist)`: gets length of list
* `range(start, stop[, step])`: returns a list containing an arithmetic progression of integers (excluding stop)
* `6 in [1,3,4]`: check if element is in list
* `[1,2,3] + [4,5,6]`: concatenate lists

## Tuples 
* `pair = (2, "a")`
* Immutable once created. `pairl[1] = "c"` will render an error.
* `pair[0]` will return 2.
* `x,y = pair` for multiple assignment

## Sets 
* `myset = {1,2,3,3}  #> set([1,2,3])`
* Unordered list
* No duplicates
* Instantiation from list: `myset = set([1,2,3,3])`
* `myset.add("something")`: adds to set (if element not in set)
* `2 in myset => True` checks for existence
* Number of elements in set `len(my_set) # 4`
* Difference, Intersection Union: `myset2 - myset`, `myset2 & myset`, `myset2 | myset`

## Dictionaries: Key-value map 
* `studentIds = {'knuth': 42.0, 'turing': 56.0, 'nash': 92.0 }`
* Getter: ` studentIds['turing']`
  * Getter with a default: `studentIds.get("foo", 0.0) # 0.0`
* Setter: `studentIds['nash'] = 'ninety-two'
* Delete a key: `del studentIds['knuth']`
* Get all keys: `mydir.keys()`
* Get all values: `mydir.values()`
* Transform to list of tuples: ` studentIds.items()`
    *  `=> [('knuth',[42.0, 'forty-two']), ('turing',56.0), ('nash','ninety-two')]`
* Length of dir: `len(mydir)`
* Key existence: ` mydir.has_key("hola")`
* Set a key if the key does not exist, else do nothing:  
```python
person = {"name": 'John'}
person.setdefault("name", "Anon")
person.setdefault("email", "not.found@example.com")
print(person) # {"name": 'John', "email", "not.found@example.com"} 
```
* DefaultDict: Same as a dictionary but never raises a `KeyError`, it is configured to return a default value at
  dictionary creation time.
```python
from collections import defaultdict

# Function to return a default values for keys that is not present
def def_value():
    return "Not Present"

# Instantiating a default dictionary. Takes in a callable function that resolved the default. Can also be a lambda
d = defaultdict(def_value)
d["foo"] # "Not Present"
```

## Iteration

```python
# List iteration
fruits = ['apples', 'oranges', 'pears', 'bananas']
for fruit in fruits:
  print(fruit + ' for sale')

# List Iteration with Index
for idx, val in enumerate(fruits):
  print(idx, val)

# Dictionary iteration
fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75}
for fruit, price in fruitPrices.items():
  if price < 2.00:
    print('%s cost %f a pound' % (fruit, price))
  else:
    print(fruit + ' are too expensive!')

# Dictionary iteration with index
# Dictionaries should not be ordered, so you shouldn't encounter this case often
for idx, (fruit, price) in enumerate(fruitPrices.items()):
  print(idx, fruit, price)
```
* break for loops with `break`
* go to next iteration with `continue`


### List Comprehensions
List comprehensions are very powerful. The example below is effectively a `map` and a `filter` combined.
```python
nums = [1,2,3,4,5,6]
oddNumsPlusOne = [x+1 for x in nums if x % 2 ==1]

# Nested list comprehension
arr = [[1,2,3], [4,5,6], [7, 8, 9]]
evens = [element for row in arr for element in row if element % 2 == 0 ]
```

## Sorting iterables
The easiest way if to use the `sorted` function. The `aList.sort()` is an older alternative available on lists.
```python
# Sorting using the "built in" element to element comparison
a = [5, 1, 4, 3]
sorted(a)  # [1, 3, 4, 5]
sorted(a, reverse=True) # [5, 4, 3, 1]

# Custom sorting by anything
# use the `key=<a_function>` to pass a function that will convert each key to a proxy value that will be used for sorting
strs = ['ccc', 'aaaa', 'd', 'bb']
sorted(strs, key=len)  ## ['d', 'bb', 'ccc', 'aaaa']  `len` here will be evaluated with each element

strs = ['xc', 'zb', 'yd' ,'wa']
def last_char(s):
    return s[-1]

sorted(strs, key=last_char) # ['wa', 'zb', 'xc', 'yd']
```

# If statements
```python
a = 200
b = 33
c = 500

if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")

# and , or 
if a > b and c > a:
  print("Both conditions are True")

# ternary
result = 'greater' if a > b else 'equal or lower'
```

# Functions
```python
def buyFruit(fruit, numPounds):
    if fruit not in fruitPrices:
        print("Sorry we don't have %s" % (fruit))
        return None # Equivalent to null
    else:
        cost = fruitPrices[fruit] * numPounds
        print("That'll be %f please" % (cost))
        return fruit # Explicit return statements

buyFruit('apples', 2.4)
```
# Classes
```python
# fruit_shop.py
class FruitShop:
    # == Class variables ==
    industry = "groceries"
    num_shops = 0
    
    # == Initialization method == 
    def __init__(self, name, fruit_prices):
        """
            name: Name of the fruit shop
            
            fruit_prices: Dictionary with keys as fruit 
            strings and prices for values e.g. 
            {'apples':2.00, 'oranges': 1.50, 'pears': 1.75} 
        """
        #  Instance variables
        self.fruit_prices = fruit_prices
        self.name = name
        self.__a_private_instance_variable = None
        
        # Accessing class variables within the instance
        # Can be accessed through the class directly or through an instance FruitShop.industry  VS myShop.industry
        # myShop.industry will first check the instance variables, then the class variables and then the superclasses'
        FruitShop.num_shops += 1 
        print(f'Welcome to the {name} fruit shop. Industry {self.industry}')

     # == Class Methods ==
    # This decorator signals that this is class method. Now we receive the class as the first argument instead 
    # of the instance.
    @classmethod 
    def set_industry(cls, new_industry):
      cls.industry = new_industry
        
    @classmethod
    # class methods can be used to create alternative constructors
    def with_default_values(cls):
      return cls("default name", {"banana": 2.0})
    
    # == Static Methods ==
    # Static methods live within the class namespace but are NOT linked directly to neither the class of an instance;
    # therefore, they do not take in the class or the instance.
    # They are used for grouping methods that are conceptually related to the class in some way
    @staticmethod
    def is_workday(day):
      if day.weekday() == 5 or day.weekday() == 6:
        return False
      return True
    
    # == Instance Methods ==
    # Instance methods have `self` as fist argument
    def get_name(self):
        return self.name
    
    def get_cost_per_pound(self, fruit):
        """
            fruit: Fruit string
        Returns cost of 'fruit', assuming 'fruit'
        is in our inventory or None otherwise
        """
        if fruit not in self.fruit_prices:
            print "Sorry we don't have %s" % (fruit)
            return None
        return self.fruit_prices[fruit]

    def get_price_of_order(self, orderList):
        """
            orderList: List of (fruit, num_pounds) tuples
            
        Returns cost of orderList. If any of the fruit are  
        """
        total_cost = 0.0
        for fruit, num_pounds in orderList:
            cost_per_pound = self.get_cost_per_pound(fruit)
            if cost_per_pound != None:
                total_cost += num_pounds * cost_per_pound
        return total_cost
    
    # == Visibility Modifiers ==
    # Enforced by a naming convention that is applicable for both variables and methods
    # `_`  means "protected". Only accessible to this class and subclasses
    # `__` means "private". Only this class can access the variables or methods
    def __sample_private_method(self):
        pass
```

## Using Classes
```python
# some_other_file.py
from fruit_shop import FruitShop # Notice how the import is declared to be able to direcly call the class
# If we used `import fruit_shop` we would have to use `fruit_shop.FruitShop(...)` to instantiate the class

sergio_shop = FruitShop("Sergio's Fruit", {'apples':2.00, 'oranges': 1.50, 'pears': 1.75})

# `self` is automatically passed as the first argument when using the `.` operator on an instance to call a method.
# Notice that these 2 are equivalent but the first does not need an argument because `self` is passed in automatically
print(sergio_shop.get_name()) # "Sergio's Fruit"
print(FruitShop.get_name(sergio_shop)) # "Sergio's Fruit".

# Using class methods (and alternative constructors)
FruitShop.set_industry("Discos")
print(FruitShop.industry)

default_shop = FruitShop.with_default_values()
print(default_shop.get_name())

# Using static methods
import datetime
my_date = datetime.date(2022, 5, 26)
print(FruitShop.is_workday(my_date))
```

# Unit Testing
* Create a sister file to the file you want to test following the name convention `test_my_file.py`
* Create a class that inherits from `unittest.TestCase`
* Each test is an instance method that begins with the name `def test_<test_description>`
* The code below shows an example of all the foundational building blocks for testing

```python
# SAMPLE CODE TO TEST
# calculator.py

def add(a, b):
    return a + b


def divide(num, denom):
    if denom == 0:
        raise ValueError("Division by zero not allowed")
    return num / denom


class Employee:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def full_name(self):
        return f'{self.first_name} {self.last_name}'

    def email(self):
        lower_first = self.first_name.lower()
        lower_last = self.last_name.lower()
        return f'{lower_first}.{lower_last}@example.com'
```

```python
# TESTING CODE
# test_calculator.py
# Note that the name of the file follows the convention `test_<file_under_test>.py`

import unittest  # The testing library. Part of the standard library
# The code I want to test
import calculator
from calculator import Employee


# Available Assertions
# =======================
# assertEqual(a, b)
# assertNotEqual(a, b)
# assertTrue(x)
# assertFalse(x)
# assertIs(a, b)  a is b checks that two objects are the same object
# assertIsNot(a, b)
# assertIsNone(x)
# assertIsNotNone(x)
# assertIn(a, b)
# assertNotIn(a, b)
# assertIsInstance(a, b)
# assertNotIsInstance(a, b)
# with self.assertRaises(MyError):

#  A testing class that MUST inherit from unittest.TestCase
class TestCalculator(unittest.TestCase):

    # Magic method that runs before each test. Must match this name
    def setUp(self):
        # Set the shared data as an instance attribute of the test class to be able
        # to access it inside tests
        self.john_employee = Employee("John", "Doe")

    # Magic method that runs after each test. Must match this name
    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("I run once before all the tests in this file")

    @classmethod
    def tearDownClass(cls):
        print("I run once after all the tests in this file")

    # This naming convention is required. A test case must be named: test_<test description>
    def test_add_works_with_positive_numbers(self):
        self.assertEqual(calculator.add(10, 5), 15)

    # A test can have multiple assertions
    def test_add_works_with_negative_numbers(self):
        self.assertEqual(calculator.add(-10, 5), -5)
        self.assertEqual(calculator.add(-10, -5), -15)

    # Assertion of raising an error
    def test_division_by_zero_not_allowed(self):
        with self.assertRaises(ValueError):
            calculator.divide(4, 0)

    def test_employee_full_name(self):
        #  john_employee is created in the setUp method
        self.assertEqual(self.john_employee.full_name(), "John Doe")

    def test_employee_email(self):
        #  john_employee is created in the setUp method
        self.assertEqual(self.john_employee.email(), "john.doe@example.com")


# This means: when this file is called as the main entrypoint of a program, run the code inside the if statement
if __name__ == '__main__':
    unittest.main()
```

# Debugging
* Pycharm debugger is great. If that fails the `print` statement is useful