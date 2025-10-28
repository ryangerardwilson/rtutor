# Python

## Part I: Python Basics

### Section 1: Dipping Toes

#### Lesson 1: Running the REPL

    $ python3
    >>> print("Hello World")
    Hello World
    >>> 1000 + 0.1 + 1 
    1001.1
    # Add 1 to the last result
    >>> _ + 1  # works in REPL only 
    1002.1
    >>> quit()  # Or Ctrl+D to exit

#### Lesson 2: Python Programs

    # hello.py
    #!/usr/bin/env python3
    print("Hello World")
    $ python3 hello.py

    # Or, make executable 
    $ chmod +x hello.py
    $ ./hello.py

#### Lesson 3: Primitives, Variables, and Expressions

    # Built-in types, considered 'primitives'. Type hints improve readability
    x: int = 42  
    y: float = 3.14159
    z: str = "Hello World"
    a: bool = True

    # An expression is anything that produces a value
    result = 2 + 3 * 4  # 14

#### Lesson 4A: Arithmetic Operators

    x = 10
    y = 3

    x + y
    x - y
    x * y
    x / y
    x // y # 3 (floor)
    x ** y # 1000
    x % y # 1
    -x # -10
    +x # 10

    # Built-in functions
    abs(-42) # 42
    divmod(10, 3) # (3, 1)
    pow(2, 3, 5) # 8 % 5 = 3
    round(3.14159, 2) # 3.14

    # Bit manipulation operators
    a = 0b11001001 # 201
    mask = 0b11110000
    (a & mask) >> 4 # 12 (0b1100)

#### Lesson 4B: Arithmetic Operators

    # Comparisons
    x == y   
    x != y  
    x < y  
    x > y 
    x <= y
    x >= y

    # Logical
    x or y
    x and y
    not x 

    # Shorthand
    x += 1
    y *= 2 

#### Lesson 5: Conditionals and Control Flow

    a = 10
    b = 5

    if a < b:
        print("Yes")
    elif a == b:
        pass  # Nada
    else:
        print("No")  

    # Conditional Expression
    maxval = a if a > b else b  # 10

    # Walrus Operator
    x = 0
    while (x := x + 1) < 10:
        print(x)  # 1 to 9

    x = 0
    while x < 10:
        x += 1
        if x == 5:
            continue
        print(x)  # 1-4,6-10

    x = 0
    while x < 10:
        if x == 5:
            break
        print(x)  # 0-4
        x += 1

#### Lesson 6A: Text Strings

    a = 'Hello World'
    b = "Python is groovy"
    c = '''Computer says no.'''  
    d = """Computer says no."""  

    # f-string
    year = 1
    principal = 1050.0
    print(f"{year:>3d}  {principal:0.2f}") # '  1  1050.00'

    # Operations
    len(a) # 11
    a[4] # 'o'
    a[-1] # 'd'
    a[:5] # 'Hello'
    a[6:] # 'World'
    a[3:8] # 'lo Wo'

    g = a.replace('Hello', 'Hello Cruel') # 'Hello Cruel World'
    g.endswith('World')   # True
    g.find('Cruel')       # 6
    g.lower()             # 'hello cruel world'
    g.split(' ')          # ['Hello', 'Cruel', 'World']
    g.startswith('Hello') # True
    g.strip()             # Trim whitespace
    g.upper()             # 'HELLO CRUEL WORLD'

#### Lesson 6B: Text Strings

    # Concat
    a + 'ly'  # 'Hello Worldly'

    # Conversion
    x = '37'
    y = '42'
    x + y               # '3742'
    int(x) + int(y)     # 79

    num = 12.34567
    str(num)                # '12.34567'
    repr('hello\nworld')    # "'hello\\nworld'"
    format(num, '0.2f')     # '12.35'
    f'{num:0.2f}'           # '12.35'

#### Lesson 7: File Input and Output

    # Read line-by-line
    with open('data.txt') as file:
        for line in file:
            print(line, end='')  # No extra \n

    # Read all
    with open('data.txt') as file:
        data = file.read()

    # Chunks with walrus
    with open('data.txt') as file:
        while (chunk := file.read(10000)):
            print(chunk, end='')

    # Write
    with open('out.txt', 'wt') as out:
        out.write(f'{year:>3d}  {principal:0.2f}\n')

    # Interactive
    name = input('Enter your name: ')
    print('Hello', name)

### Section 2: Lists, Tuples, Sets & Dicts

#### Lesson 1A: Lists

    # Lists are an ordered collection of arbitrary objects 
    empty_list = [] 
    names = ['Dave','Paula','Thomas','Lewis']
    a = [1,'Dave',3.14,['Mark',7,9,[100,101]], 10]

    # CRUD on elements
    names.append('Alex')
    names.insert(2, 'Aya') # Inserts at specific position
    a = names[2] # Indexing operator lets us access and, if required, update
    names[2] = 'Tom'

    # Slicing operator lets us access and, if required, update values across 
    # a specific index range (from x: till y)
    b = names[0:2] # ['Dave', 'Paula']
    c = names[:2] # ['Aya', 'Tom', 'Lewis', 'Alex']
    names[0:2] = ['Dave', 'Mark', 'Jeff']   

    # Looping over lists
    for name in names: print(name)

    # Concatenation                                      
    a = ['x','y'] + ['z','z','y']  

    # String to list
    letters = list('Dave')  


#### Lesson 1B: Lists

    # Performing calculations by reading data into lists. Read input lines 
    # from file of the form NAMES,SHARES,PRICE
    import sys
    if len(sys.argv) != 2:
        raise SystemExit(f'Usage: {sys.argv[0]} filename')

    rows = []
    with open(sys.argv[1], 'rt') as file:
        for line in file:
            rows.append(line.split(','))
    # rows is a list of this form [['SYM','123','456.78'], ...]

    # As a general rule, list comprehensions are a preferred technique for 
    # performing simple calculations
    total = sum([int(row[1]) * float(row[2]) for row in rows]) 
    print(f'Total cost: {total:0.2f}')

#### Lesson 2: Tuples

    # Tuples are immutable objects that help create simple data structures
    holding = ('GOOG',100,490.10)
    address = ('www.python.org',80)
    # 0- and 1-element tuples can be defined, but have special syntax:
    a = () # 0-tuple (empty tuple)
    b = (item,) # 1-tuple (note the trailing comma)

    # Accessing values
    name, shares, price = holding
    host, port = address

    # Although tuples support most of the same operations as lists (such as 
    # indexing, slicing, and concatenation), the elements of a tuple cannot be 
    # changed after creation- that is, you cannot replace, delete, or append 
    # new elements to an existing tuple. 

    # Looping over tuples
    total = 0.0
    for name, shares, prices in portfolio:
        total += shares * price
    # Alternatively:
    total = sum([shares * price for _, shares, price in portfolio]) 
    # NOTE: use _ if you don't need a value

#### Lesson 3: Sets

    # A set is an unordered collection of unique objects. Sets are used to 
    # find distinct values or to manage problems related to membership. 

    # Create a set
    empty_set = set() 
    names1 = {'IBM', 'MSFT', 'AA'}
    names2 = set(['IBM','MSFT','HPE','IBM','CAT']) # Will store 'IBM' only once
    # Elements of a set are restricted to immutable objects. You can make a set
    # of numbers, strings, or tuples - but, not of lists. You can, however, 
    create set 'from' a list, using set comprehension
    names = {s[0] for s in portfolio} 

    # Operations
    a = t | s # Union 
    b = t & s # Intersection 
    c = t - s # Difference 
    d = s - t # Difference 
    e = t ^ s # Symmetric difference: items in either s or t but not in both.

    # CRUD on elements
    t.add('DIS')
    t.update({'JJ','GE','ACME'})
    t.remove('IBM')     # Remove 'IBM' or raise KeyError if absent
    s.discard('SCOX')   # Remove 'SCOX' if it exists.

#### Lesson 4A: Dictionaries

    # A dict is a useful way to define an object that consists of named 
    # fields, and performing fast lookups on unordered data.
    prices = {} # Preferred syntax, but same as empty set, so be careful - as 
                # it may impact readability
    prices = dict() # A more explicit empty dict
    s = {
          'name' : 'GOOG',
          'shares' : 100,    
          'price' : 490
    }
    # Using tuples to create dicts with multipart keys
    prices = { }
    prices[('IBM','2015-02-03')] = 91
    prices['IBM', '2015-02-04'] = 92 # Parens can be omitted
    # NOTE: Only immutables can be used to create multipart keys, which is why 
    # we can't use lists and sets for the above

    # Value lookups
    name = s['name']
    cost = s['shares']+ s['price']

    # CRUD on elements
    s['shares'] = 75
    s['date'] = '2007-06-07'
    del s['price'] # Removes price element

#### Lesson 4B: Dictionaries

    prices = { 'GOOG' : 490, 'AAPL' : 123, 'IBM' : 91, 'MSFT' : 52 }

    # Extract value if key present 
    if 'IBM' in prices:
        p = present['IBM']
    else:
        p = 0
    # Alternatively:
    p = prices.get('IBM',0) # prices['IBM'] if it exists, else 0

    # Key extraction
    # Method 1: Convert a dict to a list
    syms = list(prices)     # syms = ['AAPL', 'MSFT', 'IBM', 'GOOG']
    # Method 2: Access via the .keys method, which returns a special dict_keys 
    # print which actively reflects changes made to the dict  
    syms = prices.keys()    # dict_keys(['GOOG', 'AAPL', 'IBM', 'MSFT'])

    # Value extraction: .values returns a special dict_values print which 
    # actively reflects changes made to the dict
    vals = prices.values()  # dict_values([490, 123, 91, 52]) 

    # Iterating over a dict in a loop
    for sym, price in prices.items():
        print(f'{sym} = {price}')

#### Lesson 4C: Dictionaries

    # Used as building blocks in data tabulation problems
    portfolio = [
        ('ACME', 50, 92.34),
        ('IBM', 75, 102.25),
        ('PHP', 40, 74.50),
        ('IBM', 40, 124.75)
    ]

    # Dict comprehension
    total_shares = { s[0]: 0 for s in portfolio }
    for name, shares, _ in portfolio:
        total_shares[name] += shares
    # total_shares = {'IBM': 125, 'ACME': 50, 'PHP': 40}

    # Alternatively: Using Counter from collections
    from collections import Counter
    total_shares = Counter()
    for name, shares, _ in portfolio:
        total_shares[name] += shares
    # total_shares = Counter({'IBM': 125, 'ACME': 50, 'PHP': 40})

### Section 3: Iteration, Looping, Functions, Exceptions & Termination

#### Lesson 1: Iteration & Looping

    # Looping over ranges: The object created by `range(i, j [,step])` is very
    # efficient for looping because it computes the values it represents on 
    # demand when lookups are requested
    a = range(1,4)      # b = 1, 2, 3    
    b = range(0, 14, 3) # c = 0, 3, 6, 9, 12
    c = range(5, 1, -1) # d = 5, 4, 3, 2

    for n in range(1,10):
        print(f'2 to the {n} power is {2**n}')

    # Looping over a string
    message = 'Hello World'
    for c in message:
        print(c)

    # Looping over a list
    names = ['Dave', 'Mark', 'Ann', 'Phil']
    for name in names:
        print(name)

    # Looping over a dict
    prices = { 'GOOG': 490, 'IBM': 91, 'AAPL': 123 }
    for key in prices
        print(key, '=', prices[key])

    # Looping over the lines of a file 
    with open('foo.txt') as file:
        for line in file:
            print(line, end='')

#### Lesson 2: Functions

    debug = True # Global variable

    # Type annotations improve readability, but are not enforced at runtime
    def remainder(int: a, int: b) -> int: 
        '''
        Computes the remainder of dividing a by b. This is a docstring, and
        feeds the help() command.
        '''
        q = a // b  
        r = a - q * b
        if debug: return r 

    result = remainder(37, 15)

    # Use a tuple to return multiple values from a function.
    def foo(a, b): 
        ...
        return (q, r)

    quotient, remainder = foo(1456, 33)

    # Assigning default values - a great way to implement 'optional features'
    def connect(hostname, port, timeout=300):
        ...

    connect('www.python.org', 80)
    connect('www.python.org', 80, 500)
    # If many defaults, readability suffers. Specify such args using keywords. 
    connect('www.python.org', 80, timeout=500)
    # Order of args doesn't matter if you know their names.
    connect(port=80, hostname='www.python.org')
