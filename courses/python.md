# Python

## Part I: Basics

### Section 1: Dipping Toes

#### Lesson 1: Running the REPL

    $ python3
    >>> print('Hello World')
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
    print('Hello World')
    $ python3 hello.py

    # Or, make executable 
    $ chmod +x hello.py
    $ ./hello.py

#### Lesson 3: Primitives, Variables, and Expressions

    # Built-in types, considered 'primitives'. Type hints improve readability
    x: int = 42  
    y: float = 3.14159
    z: str = 'Hello World'
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
        print('Yes')
    elif a == b:
        pass  # Nada
    else:
        print('No')  

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
    b = 'Python is groovy'
    c = '''Computer says no.'''  
    d = '''Computer says no.'''  

    # f-string
    year = 1
    principal = 1050.0
    print(f'{year:>3d}  {principal:0.2f}') # '  1  1050.00'

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
    repr('hello\nworld')    # ''hello\\nworld''
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

## Part II: Debugging

### Section 1: Basic Debugging

#### Lesson 1: Assertions, Logging & Raise

    # 1. Use assertions for sanity checks and try-except for graceful failures.
    #    A failed assertion raises an AssertionError('with your message')
    assert df.empty == False, 'DataFrame is empty, check your load!'
    assert 'target' in df.columns, 'Missing target column!'

    # 2. Printing everything is amateur. Use logging to track without cluttering 
    #    stdout. It is non-intrusive and persists across runs.

    import logging
    logging.basicConfig(
        level=logging.DEBUG, 
        filename='data_debug.log', 
        format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_data(df):
        logging.info(f'{df.shape}')
        try:
            cleaned = df.apply(lambda x: x * 2)  # Potential bug: non-numeric columns
            logging.debug(f'After apply: {cleaned.head()}')
        except Exception as e:
            logging.error(f'Error in processing: {e}')
            raise  
        return cleaned

    # 3. NOTE: 
    # - Use `tail -f data_debug.log`. 
    # - Levels: DEBUG for verbose, INFO for progress, WARNING/ERROR for issues. 
    # - raise the exception: This means - hey, I saw this mess, logged it for 
    #   posterity, but screw it-I'm not fixing it here; let it explode upstream 
    #   where someone smarter might handle it.

#### Lession 2: Code
   
    # 1. With code
    # Place this anywhere you want to debug
    import code; code.interact(local=locals())

#### Lesson 3A: Pdb (basics - l, p, b, q)

    # Place this anywhere you want to debug
    import pdb; pdb.set_trace()

    # Or run it like this from the start
    python -m pdb buggy_script.py

    # Or save time and drop into debugger on crash. This is fun to use with
    # assertions because a failed assertion raises an AssertionError
    python -m pdb -c continue buggy_script.py

    # Basic Commands:
    # l: list code context
    # p <var>: inspect a variable
    # b <line>: Set a breakpoint at a line number.
    # b <line>, <condition>: Conditional breakpoint: 
    # q: quit

#### Lesson 3B: Pdb (the -m flag, n and c)

    # 1. Unless run with `python -m pdb -c continue x.py`, Pdb pauses at all 
    # top level function definitions, from the top of the script to the bottom
    # - NOT as per the architectural flow of your app.  These are not break 
    # points - they are simply points at which Pdb pauses. Notice that nested 
    # functions are skipped.
    #! def main(): # pauses here
    #!     def process_numbers(numbers): # does not pause here
    #!         return total 
    #!     nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #!     process_numbers(nums)
    #!     print('Final total:', total) 

    # 2. Hitting n (next) does not guarentee us a jump to the next breakpoint. 
    #    Instead, n means 'step over' - which executes the current line, 
    #    without diving into its guts. In the below example, although we set a 
    #    breakpoint at line 5, hitting n executes everything defined at line 1, 
    #    taking us to line 14.
    #! -> def process_numbers(numbers):
    #  (Pdb) l
    #!   1  ->	def process_numbers(numbers):
    #!   2  	    total = 0
    #!   3  	    for i, num in enumerate(numbers):
    #!   4  	        if num % 2 == 0:
    #!   5  	            total += num  
    #!   6  	        else:
    #!   7  	            total -= num  
    #!   8  	        print(f'Processed {num}, total now: {total}') 
    #!  11  	    return total
    #  (Pdb) break 5
    #! Breakpoint 1 at /home/ryan/x.py:5
    #  (Pdb) n
    #! > /home/ryan/x.py(14)<module>()
    #! -> def main():

    # 3. Hitting c (continue) guarentees us a jump to the next breakpoint.
    #  (Pdb) c
    #! Processed 1, total now: -1
    #! > /home/ryan/x.py(5)process_numbers()
    #! -> total += num  # Even: add
    #  (Pdb) p total
    #! -1
    #! (Pdb)

#### Lesson 3C: Pdb (s)

    # 4. Instead of setting a breakpoint as we did above, we can hit 's' (step
    #    into) to dive into the guts on an executable line. Notice, that unlike
    #    'c' (which we can invoke from anywhere), we must be on an executable
    #    to invoke 's'. If there is no function to 'step into', 's' will behave 
    #    the same as 'n'.

    #! > /home/ryan/x.py(23)<module>()
    #! -> main()
    #  (Pdb) l
    #!  18  	    print(
    #!  19  	        'Final total:', total
    #!  20  	    )  # total is local to process_numbers, oops – but focus on debug
    #!  21
    #!  22
    #!  23  ->	main()
    #! [EOF]
    #  (Pdb) s
    #! --Call--
    #! > /home/ryan/x.py(14)main()
    #! -> def main():
    #  (Pdb) l
    #!   9  	            f'Processed {num}, total now: {total}'
    #!  10  	        )  # Line 7: We'll break here conditionally
    #!  11  	    return total
    #!  12
    #!  13
    #!  14  ->	def main():
    #!  15  	    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #!  16
    #!  17  	    process_numbers(nums)
    #!  18  	    print(
    #!  19  	        'Final total:', total
    #  (Pdb) s
    #! > /home/ryan/x.py(15)main()
    #! -> nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#### Lesson 3D: Pdb (b, run, c)

    # 5. Pdb is forward-only. You can't rewind time because pdb doesn't record 
    #    execution history. Best you can do is:
    #    - set more breakpoints with b before you screw up.
    #    - run to restart and c (continue). 
    # NOTE: While you can jump to an earlier line in the current frame (like 
    # j 42 to hop to line 42), but that's not rewinding; it's just skipping 
    # ahead or back within the same function call without re-executing side 
    # effects properly. 

    #! -> def main():
    #  (Pdb) b 5
    #! Breakpoint 1 at /home/ryan/x.py:5
    #  (Pdb) run
    #! Restarting /home/ryan/x.py with arguments:
    #! 	
    #! The program finished and will be restarted
    #! > /home/ryan/x.py(1)<module>()
    #! -> def process_numbers(numbers):
    #  (Pdb) c
    #! Processed 1, total now: -1
    #! > /home/ryan/x.py(5)process_numbers()
    #! -> total += num  # Even: add

    # NOTE: Notice that while run restarts, it does not remove the breakpoint b

## Part III: Data Science (ditch Excel)

### Section 1: Vanilla Numpy & Pandas

#### Lession 1A: Disecting a DataFrame Like a Cockroach (df.columns)

    # Structurally, a Dataframe has two main components
    df.columns
    #! Index(['id', 'to_number', 'model', 'call_type'], dtype='object')
    df.index # defaults to a range from 0 to n
    #! RangeIndex(start=0, stop=356540, step=1)

    # df.columns: may hold either an Index or a MultiIndex, and has 
    # .names and .values properties
    df.columns.names # defaults to an empty frozen list
    #! FrozenList([None])  
    df.columns.values # holds an array of tuples in case of MultiIndex
    #! array(['id', 'to_number', 'model', 'call_type'], dtype='object')

    # You can assign a name for each Index in columns. Since the above 
    # example has a single Index, we can only assign a single value
    df.column.names = ['from_employee1']
    #! from_employee1      id      to_number       model       call_type
    #! 0                    1     9999999990           z               1
    #! 1                    2     9999999991           x               2
    #! 2                    3     9999999992           a               1
    #! 3                    4     9999999993           b               2

#### Lession 1B: Disecting a DataFrame Like a Cockroach (df.index)

    # df.index: may also hold either an Index or a MultiIndex, and also
    # has .names and .values properties
    df.index.names # defaults to an empty frozen list
    #! FrozenList([None])
    df.index.values 
    #! array([   0,  1,  2, ..., 356537, 356538, 356539], shape=(356540,))

    # When we set a specific index (using set_index or groupby), 
    # - the default RangeIndex (0 to n) gets replaced by Index/MultiIndex
    # - it is added to the df.index.names 
    # - removed from df.columns (default drop=True)
    df = df.set_index(['id','model'],verify_integrity=True)
    # NOTE: Always set verify integrity as True, because it will throw an
    # error if duplicate indices are found
    df.index
    #! MultiIndex([(7937748,              '23090RA98I'),
    #!             (7938077,    'motorola edge 50 neo'),
    #!             ...
    #!             (7839768,                   'V2307')],
    #!            names=['id', 'model'], length=356540)
    df.index.names
    #! FrozenList(['id','model'])

    # Go back to default RangeIndex
    df = df.reset_index()

#### Lesson 2A: Top 11 Things to Inspect the First Time You Access a Dataframe (1-3) 

    # 1. Columns, Data types, schema, and sampling
    df.columns
    df.dtypes
    df.info() # shows both dtypes of each col and the number of non-null values
    df.shape
    df.head()
    df.col_name[row_index] # Extract cell value from table 
    df.sample(5)
    df.col_name.sample(20) 
    df.sample().T # transpose any random row
    pd.set_option('display.max_rows', 100) # You may need to pair this with the above 
                                           # if your df has an insane number of cols
    df.tail()
    print(df.to_string()) # prints all rows in a df, useful for printing
                          # grouped dfs with more than 10 rows
    df.col_name.nunique() # get count of unique values of a column
    df.col_name.unique() # get list of unique values of a column
	# See which years data we have
	df['col'] = df.col.astype('datetime64[ns]')
	df.col.dt.year.unique() 
    df.sort_values(by='col',ascending=False)

    # 2. Duplicate rows & subset
    df.duplicated().sum()
    df.duplicated(subset=['id', 'date']).sum()

    # 3. Missing values
    df.isnull().sum()
    df.isnull().mean() * 100  # % missing
    df = df[df.datetime_col.notna()] # Filter out rows with certain missing
	values
	
#### Lesson 2B: Top 11 Things to Inspect the First Time You Access a Dataframe (4-6)

    # 4. Primary key
    df.set_index(['col1','col2'], verify_integrity=True) 
    # The above will throw integrity error if the set is not a primary key. In
    # case of error, either change the set, or do: 
    df.drop_duplicates(subset=['col1','col2']) 

	# 5. Relative Distribution & Frequencies
	# For each unique value of col_a, count each unique value of col_b
	pd.crosstab(df.user_id, df.category)
	# Now to get the %distribution of values of col_b
	pd.crosstab(df.user_id, df.category, normalize='index')
	
    df.col.value_counts() # Chain: .sort_index(), .round(n), nlargest(n), nsmallest(n)
    df.groupby('col').size() # same logic as df.value_counts(), both return Series
    df.groupby(['col1','col2']).size() # works with a list of columns
	
    # 6. Summary stats - look for impossible values (e.g., negative age),
    # extreme outliers, or unexpected categories. Gives: count, unique, mean, freq, 
    # top (mode), std, min, max, quantiles
    df.describe(include='all')
    df.describe(include='all').loc['count'].T # deep dive aesthetically

#### Lesson 2C: Top 11 Things to Inspect the First Time You Access a Dataframe (7-11) 

    # 7. Percentile Analysis
    cut_off = np.percentile(df.probs, 90)
    df['meets_cutoff'] = np.where(df.probs > cut_off,1,0)
    df.meets_cutoff.values_count()

    # 8. Quantile Distribution Analysis 
    # To examine Percentile Distibution - you can use this as a histogram replacement, 
    # to check if the dist is skewed on the right or left. For instance, the below data 
    # is skewed on the right.
    df.duration_hours_col.quantile([i/10 for i in range(0,11)]).round(2)
    #! 0.0      0.06 # this is the min, or 0% of your data is below this
    #! 0.1      0.32 # 10% of your data is below this
    #! 0.2      0.73 # 20% of your data is below this  
    #! 0.3      1.43 # 30% of your data is below this 
    #! 0.4      2.32 # 40% of your data is below this 
    #! 0.5      3.50 # 50% of your data is below this 
    #! 0.6      5.27 # 60% of your data is below this 
    #! 0.7      9.34 # 70% of your data is below this 
    #! 0.8     23.16 # 80% of your data is below this 
    #! 0.9     28.94 # 90% of your data is below this 
    #! 1.0    312.47 # this is the max, or 100% of your data is below this
    #! Name: duration_hours_col, dtype: float64

    # 9. Correlations & Multicollinearity
    corr_matrix = df.corr(numeric_only=True)  

    # 10. Domain Consistency & Business Logic Checks
    assert (df.age_col >= 0).all(), 'Negative ages found!'

    # 11. Quick filteration / masking based analysis
    row_condition = df.assigned_col.notna()
    df[['mobile', 'account_id', 'assigned', 'otp']][row_condition]

#### Lesson 3: Impact of Scope on Dataframe Mutations

    # Never mutate a DataFrame that came from outside your function. The below 
    # will throw a SettingWithCopyWarning, as pandas will enforce copy() under 
    # the hood, and throw an ugly warning
    #! def mutate_df(df, target):
    #!     df['target'] = df.price.astype(str).replace('$','').astype(float)
    #!     return df

    # Instead, do this, when you want to use a function to mutate a df
    def mutate_df(df, target):
        df = df.copy()
        df['target'] = df.price.astype(str).replace('$','').astype(float)
        return df

#### Lesson 4: Modifications / Cleaning Based on Initial Inspection 

    df.info()

    # 1. Cleaning rows
    # Filter out rows where a specific col has null values
    df = df[df.datetime_col.notna()] 
    # Filtering rows and columns in one line
    df[['mobile', 'account_id', 'assigned', 'otp']][df['assigned'].notna()]

    # 2. Cleaning column names
    # Rename specific columns 
    df = df.rename(columns={'old_name1': 'new_name1', 'old_name2': 'new_name2'})
	# Lowercase all column names
	df.columns = df.columns.str.lower()

    # 3. Cleaning object dtype column values
    df['col'] = df.col.str.replace('$'.'').str.replace(','.'')
    df['col'] = df.col.map({'t':True, 'f':False})

    # 4. Converting object dtypes to int, float, bool
    # Works if the column has no non-numeric values
    # int32 uses 4byts with range -2bn to +2bn, int64 uses 2x bytes giving range -9.2 
    # quintillion to + 9.2 quintillion. We generally use int64 because int32
    # isn't nullable
    df['col'] = df.col.astype('int64') 
    # float32 is good enough for most use cases; uses 4bytes, same range as int 32, but 
    # with 7-8 decimals, float64 uses 2x bytes, same range as int64, with 8-15 decimals 
    df['col'] = df.col.astype('float32') 
    df['col'] = df.col.astype('datetime64[ns]')
    df['col'] = df.col.astype('timedelta64[ns]')
    df['col'] = df.col.astype('category') # Speeds up computations; works when your 
                                          # column is a dumpsterfire of non-numeric mixed dtypes as well
    # Use coercion if you want to treat the non-parseable values of your column as nan/nat/empty 
    df['col'] = pd.to_numeric(df.col, errors='coerce').astype('int64')
    df['col'] = pd.to_numeric(df.col, errors='coerce').astype('float64')
    df['col'] = pd.to_datetime(df.col, errors='coerce')  
    df['col'] = pd.to_datetime(df.col, errors='coerce', unit='ms') # Handles unix values

#### Lesson 5: Filtering 

    # Dot notation v. [] notation
    df[df.price > 0] # Pandas lets you use dot notation if your column name is a valid variable name

    # The correct way to store filtered dfs
    # This confuses pandas, on whether to return a view or a do an assignment
    df = df[df['price'] != 0] # Wrong approach - will return a SettingWithCopyWarning
    # Correct approaches
    df = df[df.price != 0].copy()
    df2 = df[df.price != 0] 

    # Boolean filter
    df[((df.plan_duration > 12) & (df.status.isin([6,12,24]))) | (df.plan_type == 'promo')]

    # Exclusion
    df[~df.plan_id.isin([4, 5])]

    # isna and notna
    df[df.mac.isna()]
    df[df.mac.notna()]

    # Mask example
    mask = (df.plan_duration > 12) & (df.plan_id == 3)
    df.loc[mask, ['mac', 'mobile', 'plan_id']]
    df.loc[mask, 'plan_duration'] = 0

    # String filters
    df[df.mobile.str.contains('555', na=False)]
    df[df.mac.str.startswith('aa', na=False)]

    # Mutate/ Copy
    df = df[df.plan_duration > 12]
    filtered_df = df[df.plan_duration > 12].copy()

    # Range comparisons / between - readable and vectorized
    df[df.ts >= pd.Timestamp('2020-01-01')] # single-side
    df[df.ts.between('2020-01-01', '2020-01-31')] # inclusive range (clean)

    # Component masks with .dt - year/month/weekday/time
    df[df.ts.dt.year == 2020] # filter by year
    df[df.ts.dt.month.isin([1,2,3])] # filter months
    df[df.ts.dt.weekday < 5] # weekday mask (Mon=0)
    df[df.ts.dt.time.between(pd.to_datetime('08:00').time(), pd.to_datetime('17:00').time())] # time-only mask

    # Filtering rows and columns in one line
    df[['mobile', 'account_id', 'assigned', 'otp']][df.assigned.notna()]

#### Lesson 6: Using Python to Implement the Relational Model

    # A table/dataframe is a way to represent an n-ary mathematical relation, where
    # - n represents the number of columns,
    # - columns represent attributes of tuple indices, and
	# - rows represent a set of {tuples}.
    # Table = { (c1, c2, ..., cn) | each ci in `domain_i`, for i=1 to n }

    # Define the domains implicitly through data types and values
    #! table = {
    #!     'ID': [1, 2, 3, 3], # Domain: positive integers
    #!     'Name': ['Alice', 'Bob', 'Charlie', 'Charlie'], # Domain: strings
    #!     'Salary': [100000.0, 120000.0, 90000.0, 90000.0] # Domain: non-negative floats
    #! }

    # While a df, can accomodate duplicate rows - we cannot call such a table a
    # relational table because, rows MUST represent a set of {tuples}. A
    # df representing a relation must have at least 1 key candidate.
    df = pd.DataFrame(table)
    df.set_index('id',verify_integrity=True) 
    # NOTE: In case of integrity error, do df.drop_duplicates() or
    # df.drop_duplicates(subset=[list_of_keys_to_be_indexed_upon])
    n = len(df.columns)
    columns = df.columns

#### Lesson 7: Indexing Advantages 

    #!    employee_id   department  hire_date     name  salary
    #! 0          101           HR 2023-01-01    Alice   60000
    #! 1          102  Engineering 2023-01-04      Bob   80000
    #! 2          101  Engineering 2023-01-02  Charlie   75000
    #! 3          103        Sales 2023-01-03    David   70000
    df = df.set_index(['employee_id', 'department', 'hire_date'],verify_integrity=True)

    # 1. Easily sort rows in the order of the index
    df = df.sort_index()  
    #!                                        name  salary
    #! employee_id department  hire_date
    #! 101         Engineering 2023-01-02  Charlie   75000
    #!             HR          2023-01-01    Alice   60000
    #! 102         Engineering 2023-01-04      Bob   80000
    #! 103         Sales       2023-01-03    David   70000

    # 2. Fast lookups. Grab the row for employee 101 in Engineering on 2023-01-02
    df.loc[(101, 'Engineering', '2023-01-02')]
    #! name      Charlie
    #! salary      75000

    # 3. Datetime index slicing by temporarily setting hire_date as the single index
    temp_df = df.reset_index().set_index('hire_date',verify_integrity=True).sort_index().loc['2023-01-01':'2023-01-03']
    #!             employee_id   department     name  salary
    #! hire_date
    #! 2023-01-01          101           HR    Alice   60000
    #! 2023-01-02          101  Engineering  Charlie   75000
    #! 2023-01-03          103        Sales    David   70000

    # 4. Partial string slicing (e.g., all of January 2023)
    temp_df.loc['2023-01']
    #!             employee_id   department     name  salary
    #! hire_date
    #! 2023-01-01          101           HR    Alice   60000
    #! 2023-01-02          101  Engineering  Charlie   75000

#### Lesson 8A: Joins (union join aka full outer join)

    #! print(df, other_df)
    #!                             name  salary
    #! employee_id department
    #! 101         HR             Alice   60000
    #! 102         Engineering      Bob   80000
    #! 101         Engineering  Charlie   75000
    #! 103         Sales          David   70000
    #!
    #!                          bonus
    #! employee_id department
    #! 101         HR            5000
    #!             Engineering  10000
    #! 104         Marketing    12000

    union_joined_df = df.join(other_df, how='outer')
    #!                             name   salary    bonus
    #! employee_id department
    #! 101         Engineering  Charlie  75000.0  10000.0
    #!             HR             Alice  60000.0   5000.0
    #! 102         Engineering      Bob  80000.0      NaN
    #! 103         Sales          David  70000.0      NaN
    #! 104         Marketing        NaN      NaN  12000.0

#### Lesson 8B: Joins (left join)

    #! print(df, other_df)
    #!                             name  salary
    #! employee_id department
    #! 101         HR             Alice   60000
    #! 102         Engineering      Bob   80000
    #! 101         Engineering  Charlie   75000
    #! 103         Sales          David   70000
    #!
    #!                          bonus
    #! employee_id department
    #! 101         HR            5000
    #!             Engineering  10000
    #! 104         Marketing    12000

    # Keep all from left df, add matches from right
    left_joined_df = df.join(other_df, how='left')
    #!                             name  salary    bonus
    #! employee_id department
    #! 101         HR             Alice   60000   5000.0
    #! 102         Engineering      Bob   80000      NaN
    #! 101         Engineering  Charlie   75000  10000.0
    #! 103         Sales          David   70000      NaN

#### Lesson 8C: Joins (inner join)

    #! print(df, other_df)
    #!                             name  salary
    #! employee_id department
    #! 101         HR             Alice   60000
    #! 102         Engineering      Bob   80000
    #! 101         Engineering  Charlie   75000
    #! 103         Sales          David   70000
    #!
    #!                          bonus
    #! employee_id department
    #! 101         HR            5000
    #!             Engineering  10000
    #! 104         Marketing    12000

    # Only where keys overlap in both
    inner_joined_df = df.join(other_df, how='inner')
    #!                             name  salary  bonus
    #! employee_id department
    #! 101         HR             Alice   60000   5000
    #!             Engineering  Charlie   75000  10000

#### Lesson 9A: groupby (unindexed dataframe) 

    #!    plan_id mobile   mac  value
    #! 0        1      A  mac1     10
    #! 1        1      A  mac2     20
    #! 2        1      B  mac3     30
    #! 3        2      B  mac4     40
    #! 4        2      C  mac5     50
    #! 5        3      C  mac6     60

    # Group by column
    df = df.groupby(['plan_id', 'mobile']).agg(
        nmbr_mac=('mac', 'count'),
        # other aggs, if any
    )
    #!                 nmber_mac
    #! plan_id mobile
    #! 1       A               2
    #!         B               1
    #! 2       B               1
    #!         C               1
    #! 3       C               1

    # NOTE: Available aggs -> count, size, nunique, min, max, first, last, sum, mean, 
    # median, mode

    #! How data structure is altered?
    #! df.index.names     # FrozenList([None])  ->  FrozenList(['plan_id', 'mobile'])
    #! df.index.values    # array([0,1,...,n])  ->  array[(1,'A'),(1,'B'),...,(3,'C')
    #! df.columns.names   # no change, remains FrozenList([None])
    #! df.columns.values  # array(['plan_id',...,'value'])  -> array(['nmbr_mac'])

#### Lesson 9B: groupby (indexed dataframe) 

    #!    plan_id mobile   mac  value
    #! 0        1      A  mac1     10
    #! 1        1      A  mac2     20
    #! 2        1      B  mac3     30
    #! 3        2      B  mac4     40
    #! 4        2      C  mac5     50
    #! 5        3      C  mac6     60

    df = df.set_index(['plan_id', 'mobile','mac'],verify_integrity=True)
    #!    plan_id mobile   mac  value
    #! 0        1      A  mac1     10
    #! 1        1      A  mac2     20
    #! 2        1      B  mac3     30
    #! 3        2      B  mac4     40
    #! 4        2      C  mac5     50
    #! 5        3      C  mac6     60

    # Same syntax as before, except we specify the level param
    df = df.groupby(level=['plan_id', 'mobile']).agg(
        nmbr_mac=('value', 'count'),  
        # other aggs
    )
    #!                 nmbr_mac  
    #! plan_id mobile
    #! 1       A              2 
    #!         B              1
    #! 2       B              1
    #!         C              1
    #! 3       C              1

    #! How data structure is altered?
    #! df.index.names     # no change, remains FrozenList(['plan_id','mobile']) 
    #! df.index.values    # no chnage, remains array[(1,'A'),(1,'B'),...,(3,'C') 
    #! df.columns.names   # no change, remains FrozenList([None])
    #! df.columns.values  # array(['mac','value'])  ->  array(['nmbr_mac']) 

#### Lesson 9C: groupby (shorthand syntax) 

    df.groupby('plan_id').size()    # counts all rows; same logic as df.value_counts(), 
                                    # thus returns a Series. chain .to_frame(name='count') 
                                    # to output a df instead.

    # The below return dfs
    df.groupby(level=['plan_id', 'mobile'])[['usage_gb']].mean()
    df.groupby(level=['plan_id', 'mobile'])[['usage_gb', 'cost']].mean()
    df.groupby(level=['plan_id', 'mobile']).mean(numeric_only=True)

    df.groupby('plan_id').count() # counts rows, but skips NaNs

    # NOTE: Available aggs -> count, size, nunique, min, max, first, last, sum, 
    # mean, median, mode

#### Lesson 10A: Feature Engineering (Create helper columns with piped functions)

    #! df
    #!   verification_methods  price  reviews
    #! 0          email,phone    100      4.5
    #! 1                email    150      3.8
    #! 2     phone,work_email    200      4.9

    def generate_vcount(df, feature_name):
        # axis=1 iterates row by row
        def compute_vcount(row):
            verif_count = len(row['verification_methods'].split(','))
            adjusted = row['price'] / row['reviews'] if row['reviews'] > 0 else 0
            return verif_count * adjusted

        df[feature_name] = df.apply(compute_vcount, axis=1)
        return df

    def generate_discounted_price(df):
        df['discounted_price'] = df['price'] * 0.9  
        return df

    def generate_n_price(df):
        if 'price' in df.columns:
            min_val = df.price.min()
            max_val = df.price.max()
            if max_val > min_val:  
                df.n_price = (df.price - min_val) / (max_val - min_val)
            else:
                df.n_price = 0.0  
        return df

    # This allows us to comment out features by line, and neatly encapsalte complexities,
    # and also control the feature names if we want
    df = (
        df.pipe(generate_vcount, 'fv_count')
        .pipe(generate_discounted_price)
        .pipe(generate_n_price)
    )
    # Retain only features
    df[['fv_count', 'discounted_price', 'n_price']]
    #!     fv_count  discounted_price  n_price
    #! 0  44.444444              90.0      0.0
    #! 1  39.473684             135.0      0.5
    #! 2  81.632653             180.0      1.0

#### Lesson 10B: Feature Engineering (Creating helper bin classification and category columns)

    #! df
    #!    utilisation  number_days    id   otp
    #! 0         0.75           15  1234  None
    #! 1         0.95           28  None  ABCD
    #! 2         0.85           40  5678  EFGH

    def generate_util_range_qbc(df):
        # Uses pd.qcut for quantile / equal-frequency bins cut by data quantiles so bins have 
        # ~equal counts. We drop duplicates to merge duplicate bins caused by too many duplicate values.
        df['util_range_qbc'] = pd.qcut(df['utilisation'], q=10, duplicates='drop', labels=False)
        return df

    def generate_days_rng_bc(df):
        # NOTE: labels=False gives us the index number of the label (which can directly be used as a 
        # numeric feature), instead of the label itself. Don't add this param if you want the col more readable
        df['days_rng_bc'] = pd.cut(df['number_days'], bins=[0, 10, 20, 28, 35, float('inf')], labels=False)
        return df

    def generate_cohort(df):
        conditions = [
            (df['id'].notna() & df['otp'].isna()),
            (df['id'].isna() & df['otp'].notna()),
            (df['id'].notna() & df['otp'].notna()),
            (df['id'].isna() & df['otp'].isna()),
        ]
        choices = ['CALL_NOINSTALL', 'NOCALL_INSTALL', 'CALL_INSTALL', 'NOCALL_NOINSTALL']
        df['cohort'] = np.select(conditions, choices, default=None)
        return df

    df = (
        df.pipe(generate_util_range_qbc)
        .pipe(generate_days_rng_bc)
        .pipe(generate_cohort)
    )
    # Retain only features
    df[['util_range_qbc', 'days_rng_bc', 'cohort']]
    #!    util_range_qbc  days_rng_bc         cohort
    #! 0               0            1  CALL_NOINSTALL
    #! 1               2            2  NOCALL_INSTALL
    #! 2               1            4    CALL_INSTALL

#### Lesson 10C: Feature Engineering (Creating helper computation and boolean columns)

    #! df
    #!    number_days  plan_duration       start_timestamp         end_timestamp
    #! 0           15             30  2025-11-01 10:00:00  2025-11-01 10:30:00
    #! 1           28             30  2025-11-02 14:00:00  2025-11-02 15:45:00
    #! 2           40             45  2025-11-03 09:00:00  2025-11-03 09:20:00

    def generate_utilisation(df):
        df['utilisation'] = df['number_days'] / df['plan_duration']
        return df

    def generate_diff_mins(df):
        df['diff_mins'] = (df['end_timestamp'] - df['start_timestamp']).dt.total_seconds() / 60
        return df

    def generate_mac_90(df):
        df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0)
        return df

    def generate_mac_80(df):
        df['mac_80%'] = np.where((df['utilisation'] > 0.8) & (df['utilisation'] <= 0.9), 1, 0)
        return df

    # Pipe 'em all—comment out lines if you want to disable a feature without breaking the chain.
    df = (
        df.pipe(generate_utilisation)
        .pipe(generate_diff_mins)
        .pipe(generate_mac_90)
        .pipe(generate_mac_80)
    )
    # Retain only features
    df[['utilisation', 'diff_mins', 'mac_90%', 'mac_80%']]
    #!   utilisation  diff_mins  mac_90%   mac_80%
    #! 0    0.500000       30.0        0         0
    #! 1    0.933333      105.0        1         0
    #! 2    0.888889       20.0        0         1

#### Lesson 11A: Pivot (single index and multi index)

    # We don't really need the .pivot and .pivot_table methods to pivot a
    # dataframe. This is because, the below mathematical definition of a pivot
    # table, makes it possible by simply unstacking a grouped aggregate
    # DEF: Given a relation (table) T with attributes {R_attrs} (row keys), 
    # {C_attrs} (column keys), and {V} (value(s)), and an aggregation function 
    # agg, the pivot table P is the function:
    #   P(r, c) = agg({ t.V | t in T and t.R_attrs = r and t.C_attrs = c }),
    #   where r ranges over unique values of R_attrs and c over unique values
    #   of C_attrs
    #!    foo bar  baz
    #! 0  one   A    1
    #! 1  one   B    2
    #! 2  one   A    5
    #! 3  two   A    3
    #! 4  two   B    4

    single_index_pivot = df.groupby('foo').agg(baz_sum=('baz', 'sum'))
    #!      baz_sum
    #! foo
    #! one        8
    #! two        7

    multi_index_pivot = df.groupby(['foo','bar']).agg(baz_sum=('baz', 'sum'))
    #!          baz_sum
    #! foo bar
    #! one A          6
    #!     B          2
    #! two A          3
    #!     B          4

#### Lesson 11B: Pivot (`pivot_table`)

    # Now, we use the df.pivot_table method to achieve the same results as the previous lesson
    #!    foo bar  baz
    #! 0  one   A    1
    #! 1  one   B    2
    #! 2  one   A    5
    #! 3  two   A    3
    #! 4  two   B    4

    single_index_pivot = df.pivot_table(index='foo', values='baz', aggfunc='sum')
    #!      baz_sum
    #! foo
    #! one        8
    #! two        7

    multi_index_pivot = df.pivot_table(index=['foo', 'bar'], values='baz', aggfunc='sum').rename(columns={'baz':'baz_sum'})
    #!          baz_sum
    #! foo bar
    #! one A          6
    #!     B          2
    #! two A          3
    #!     B          4

#### Lesson 11C: Pivot (flattening a multi index)

    #! print(multi_index_pivot, multi_index_pivot.columns)
    #!          baz_sum
    #! foo bar
    #! one A          6
    #!     B          2
    #! two A          3
    #!     B          4
    #! Index(['baz_sum'], dtype='object')

    # While the above lays out a neat looking hierachial tree, it is useful to
    # 'flatten' it using unstack as below.
    multi_index_pivot = multi_index_pivot.unstack()
    #!     baz_sum
    #! bar       A  B
    #! foo
    #! one       6  2
    #! two       3  4
    #! MultiIndex([('baz_sum', 'A'), ('baz_sum', 'B')], names=[None, 'bar'])

    # We can further flatten the unstacked df from MultiIndex columns to Index
    # columns as below 
    multi_index_pivot.columns = multi_index_pivot.columns.droplevel(0)
    print(multi_index_pivot)
    #!              A          B
    #! foo
    #! one          6          2
    #! two          3          4

#### Lesson 12A: Rank Ordering Basics

    # Rank ordering is simply UI in data science, to present data in a way
    # business can digest

    # Presenting data like this will give business a headache
    #!   item  score1  score2   value
    #! 0    A       1      10      10
    #! 1    B       3       8      20
    #! 2    C       5       6      30
    #! 3    D       7       4      40
    #! 4    E       9       2      50
    #! 5    F       2       1      60
    #! 6    G       4       3      70
    #! 7    H       6       5      80
    #! 8    I       8       7      90
    #! 9    J      10       9     100

    # But, this will give them an orgasm
    #!    level1  level2  count  total_value  avg_value
    #! 0   1_low   1_low      1           60       60.0
    #! 0   1_low   2_med      1           60       60.0
    #! 1   1_low  3_high      2           30       15.0
    #! 2   2_med   1_low      1           70       70.0
    #! 3   2_med   2_med      2          110       55.0
    #! 3   2_med   3_high     2          110       55.0
    #! 4  3_high   1_low      1           50       50.0
    #! 5  3_high   2_med      1           40       40.0
    #! 6  3_high  3_high      2          190       95.0

#### Lesson 12B: Rank Ordering nD Data (computation)

    #!   item  score1  score2  score3  value
    #! 0    A       1      10       4     10
    #! 1    B       3       8       5     20
    #! 2    C       5       6       6     30
    #! 3    D       7       4       7     40
    #! 4    E       9       2       8     50
    #! 5    F       2       1       9     60
    #! 6    G       4       3      10     70
    #! 7    H       6       5       1     80
    #! 8    I       8       7       2     90
    #! 9    J      10       9       3    100

    # The simplest and most efficient way to rank order is to convert bins to
    levels, and then, group by levels
    df['level1'] = np.where(df['score1'] >= 7, '3_high', np.where(df['score1'] >= 4, '2_med', '1_low'))
    df['level2'] = np.where(df['score2'] >= 7, '3_high', np.where(df['score2'] >= 4, '2_med', '1_low'))
    df['level3'] = np.where(df['score3'] >= 7, '3_high', np.where(df['score3'] >= 4, '2_med', '1_low'))
    grouped = df.groupby(['level1', 'level2', 'level3']).agg(
        count=('item', 'nunique'), total_value=('value', 'sum'), avg_value=('value', 'mean')
    )
    grouped.sort_values(['level1', 'level2', 'level3'])
    print(grouped)
    #!                       count  total_value  avg_value
    #! level1 level2 level3
    #! 1_low  1_low  3_high      1           60       60.0
    #!        3_high 2_med       2           30       15.0
    #! 2_med  1_low  3_high      1           70       70.0
    #!        2_med  1_low       1           80       80.0
    #!               2_med       1           30       30.0
    #! 3_high 1_low  3_high      1           50       50.0
    #!        2_med  3_high      1           40       40.0
    #!        3_high 1_low       2          190       95.0


#### Lesson 12C: Rank Ordering 2D Data (Motivation x Ability Grid)

    # If the focus of our rank ordering is 2D with only one item to aggregate,
    # we could also present the rank ordering in grid. For instance a 3x3 grid 
    # of ability x motivation, aggregating the count of unique users.

    # Let's consider a df which has the columns:
    # - util_rng_qc: 1-10 (utilisation quantile, 10 = highest usage)
    # - churn_risk_qc: 1-10 (churn risk quantile, 10 = highest risk)

    df['motivation'] = np.where(
        df['churn_risk_qc'] <= 3,
        '3_high',
        np.where(df['churn_risk_qc'] <= 7, '2_med', '1_low'),
    )
    df['ability'] = np.where(
        df['util_rng_qc'] >= 9,
        '3_high',
        np.where(df['util_rng_qc'] >= 4, '2_med', '1_low'),
    )

    pk_motivation_df = (
        df.groupby(['motivation', 'ability'])
        .agg(users=('plan_id', 'nunique'))
        .unstack()
        .fillna(0)
    )
    # NOTE:
    # - We wrap it in a (), to allow us to indent each .method on a seperate line
    # - We do NOT need to reset index here, since we'll use unstack instead of pivot

    pk_motivation_df.columns = pk_motivation_df.columns.droplevel(0)
    print(pk_motivation_df)
    #! ability     1_low  2_med  3_high
    #! motivation
    #! 1_low         4.0    0.0     0.0
    #! 2_med         1.0    5.0     0.0
    #! 3_high        0.0    1.0     4.0

#### Lesson 13A: Subset Aggregation / SQL sub-queries (using groupby)

    #!    mobile  account_id    event_name  added_time
    #! 0  123456           1      ASSIGNED          10
    #! 1  123456           1  OTP_VERIFIED          15
    #! 2  123456           1   OTHER_EVENT          20
    #! 3  789012           2      ASSIGNED           5
    #! 4  789012           2      ASSIGNED           3
    #! 5  345678           3  OTP_VERIFIED           8
    #! 6  901234           4      ASSIGNED          12

    # Get min ASSIGNED time per mobile and account_id as a DataFrame with MultiIndex
    assigned_df = (
        df[df['event_name'] == 'ASSIGNED']
        .groupby(['mobile', 'account_id'])
        .agg(assigned_added_time=('added_time', 'min'))
    )
    #!    mobile  account_id  assigned_added_time
    #! 0  123456           1                   10
    #! 1  789012           2                    3
    #! 2  901234           4                   12

    # Get min OTP_VERIFIED time per mobile and account_id 
    otp_df = (
        df[df['event_name'] == 'OTP_VERIFIED']
        .groupby(['mobile', 'account_id'])
        .agg(otpv_added_time=('added_time', 'min'))
    )
    #!    mobile  account_id  otpv_added_time
    #! 0  123456           1               15
    #! 1  345678           3                8
    
    # Merge on the index; keep all from left, matches from right, NaN where no OTP
    otp_info = assigned_df.merge(otp_df, on=['mobile', 'account_id'], how='left')

    #!    mobile  account_id  assigned_added_time  otpv_added_time
    #! 0  123456           1                   10             15.0
    #! 1  789012           2                    3              NaN
    #! 2  901234           4                   12              NaN

#### Lesson 13B: Subset Aggregation / SQL sub-queries (using pivot)

    #!    mobile  account_id    event_name  added_time
    #! 0  123456           1      ASSIGNED          10
    #! 1  123456           1  OTP_VERIFIED          15
    #! 2  123456           1   OTHER_EVENT          20
    #! 3  789012           2      ASSIGNED           5
    #! 4  789012           2      ASSIGNED           3
    #! 5  345678           3  OTP_VERIFIED           8
    #! 6  901234           4      ASSIGNED          12

    pivot_df = df_tasks.pivot_table(
        index=['mobile', 'account_id'],
        columns='event_name',
        values='added_time',
        aggfunc='min'
    ).reset_index()
    pivot_df.columns.name = None

    #! event_name  mobile  account_id  ASSIGNED  OTHER_EVENT  OTP_VERIFIED
    #! 0           123456           1      10.0         20.0          15.0
    #! 1           345678           3       NaN          NaN           8.0
    #! 2           789012           2       3.0          NaN           NaN
    #! 3           901234           4      12.0          NaN           NaN

    # Then slice to just the columns you care about, renaming if needed
    otp_info = (
        pivot_df[['mobile', 'account_id', 'ASSIGNED', 'OTP_VERIFIED']][pivot_df['ASSIGNED'].notna()]
        .rename(columns={'ASSIGNED': 'assigned_added_time', 'OTP_VERIFIED': 'otpv_added_time'})
    )

    #!    mobile  account_id  assigned_added_time  otpv_added_time
    #! 0  123456           1                   10             15.0
    #! 1  789012           2                    3              NaN
    #! 2  901234           4                   12              NaN

### Section 2: Machine Learning (XGBoost)

#### Lesson 1A: Data Visualization Fundamentals (Normalised vs. De-normalised data)

	# Normalization (in the original Codd/1970 relational database sense) =
	# organizing tables to eliminate redundancy and update anomalies by
	# splitting data into many narrow, tightly linked tables following strict
	# normal forms

	# Core Idea: Every piece of information appears exactly once, and everything else
	# references it via keys.

	#! Real-world example - fully normalized version of the same data:
	#! Table	   		   Key 	   	 	   Columns
	#! customers  		   customer_id	   customer_id, name, birth_date, city
	#! orders	   		   order_id		   order_id, customer_id, order_date, amount
	#! order_items		   item_id		   item_id, order_id, product_id, qty, price
	#! products		   product_id	   product_id, product_name, category
	#! payments		   payment_id	   payment_id, order_id, payment_date, method

	# Why Normalization Is the Enemy of Real-World ML? To train a model you
	# would have to execute massive star-schema joins + window aggregations
	# on-the-fly for every single prediction or training row. That's 100-1000x
	# slower and leaks like crazy if you're not extremely careful

#### Lesson 1B: Data Visualization Fundamentals (Mathematical Data Structures)

	# Generally speaking, data appears in the real world in the following
	# mathematical forms:
	# a. Event/ logs: they simply record a series of events around a time axis
	# b. Real world Tabular Data: a denormalized, point-in-time feature matrix. Note
	#    that this is not a relation, but a matrix, which is a superset of a relation.
	# c. Codd's original 1970 relational model - a time-varying relation that holds
	#    only currently valid tuples

	# What is a good data set to model?
	# A good dataset for supervised learning is a point-in-time feature table, rooted in the REAL WORLD,
	# meeting the following conditions:
	# - Each row corresponds to a unique real world entity (e.g., customer, vehicle, patient, etc.)
	#   observed at a specific prediction timestamp t_pred, collectively, creating a 'Monte-Carlo'
	#   sample (Whats a Monte-Carlo sample? Let reality roll the dice for you, then write down what
	#   actually happened. You repeat that many times, and you get a bunch of real-world random samples.
	# - The target variable y_i is the outcome of interest that materializes strictly after t_pred
	#   (e.g., failure within next 30 days, churn in the following month, etc.)
	# - For every feature x_i, the value j in the row must be known or computable using only
	#   information available at or before t_pred. No feature may incorporate data from any time > t_pred
	#   (this prohibition is called 'future leakage' or 'target leakage')
	# - There are no duplicate rows that are invalid Monte-Carlo samples
	
	# Why do 'good' duplicates (i.e. duplicates that are valid Monte-Carlo samples) actually add value?
	# - They teach the model the actual odds: when everything looks exactly like this, it fails x% of the time.
	# - If you delete all of them and keep only one, the model thinks 'this never happens' or 'this always
	#   happens', causing totally wrong probabilities.
	# - Every extra copy is like flipping the real-world coin one more time and writing down what actually happened.

#### Lesson 1C: Using Definitional Precision to Prevent Target Leakage

	# Now, the following definitions become clear:
	# Feature: Anything known ON_OR_BEFORE prediction_time
	# Target: A value that is realized AFTER prediction_time. For instance,
	# - for linear regression: number OR log(number) AFTER {prediction_time}
	# - for binary classification: 1 IF {condition} AFTER {prediction_time} ELSE 0

	# To understand the importance of the above definitions, lets say we define our
	# target as follows: WITH {prediction_time}: 1 IF {condition} ELSE 0
	# The problem that is created is called 'target leakage', where the rule
	# governing the impact of prediction_time on the outcome is ambiguously
	# stated.

	# Thus, in preprocessing raw data, it is imperative to ensure that the data
	# filtered such that:
	# - target is AFTER prediction_time
	# - all features exist ON_OR_BEFORE prediction_time
	
#### Lesson 2: Decision Trees

    # A decision tree is like a flowchart for making predictions. It starts at the 
    # root (top question), splits into branches based on features (e.g., 'Is the 
    # house in a fancy neighborhood?'), and ends at leaves-the terminal nodes where 
    # the prediction value lives. Those leaf values (often averages or counts from 
    # your data) are basically the 'weights' that determine the output for whatever 
    # path you took.

    #! Evolution of Decision Tree Algos                                                                     
    #! -----------------------------------------------------------------------------------------------------|
    #!  year |           algo |                                                                      impact |
    #! -----------------------------------------------------------------------------------------------------|
    #!  1963 |            AID |                              First automated binary splitting (stats-based) |
    #!  1980 |          CHAID |                                      Chi-squared tests for multi-way splits |
    #!  1984 |           CART |                           handles missing data; foundation for modern trees |
    #!  1986 |            ID3 |                                       Information gain (entropy) for splits |
    #!  1993 |           C4.5 |                     Improved ID3: pruning, continuous vars, rule extraction |
    #!  1995 |       AdaBoost | First popular boosting ensemble for trees, adaptively weights weak learners |
    #!  2001 |            GBM |                    Gradient descent on trees for better predictive accuracy |
    #!  2001 | Random Forests |          Ensembles of trees (bagging + random features) for better accuracy |
    #!  2014 |        XGBoost |   Scalable, regularized gradient boosting with tree pruning and parallelism |
    #! -----------------------------------------------------------------------------------------------------|

#### Lesson 3A: XGBoost Intuition (gradient boosting, overfitting, regularization)

    # 1. Gradient Boosting
    # Think of gradient boosting like building a team of weaklings who, together, 
    # kick ass:
    # - You start with one mediocre decision tree (a 'weak learner' - a tree with 
    #   2 or 3 branches) that tries to predict your target (say, house prices 
    #   based on features like size and location). It sucks, makes big errors. 
    # - Then, you train another tree focused only on fixing those errors-like a 
    #   specialist patching up the mistakes. 
    # - Repeat this: each new tree boosts the overall model by targeting the 
    #   residuals (what's left unexplained). 

    # 2. Overfitting & Regularization
    # XGBoost amps up gradient boosting with regularization to stop overfitting. 
    # - Overfitting: Picture your model as a student cramming for a test. If he 
    #   memorizes every tiny detail from the study notes (your training data), he 
    #   aces that but bombs on anything new (like real-world test data). 
    # - Regularization: A safety net that adds rules to keep the  model from 
    #   becoming too complex. 
    # NOTE: XGBoost uses L2 Regularization: It's named after the L2 norm, which is 
    # math-speak for the Euclidean distance thing-basically, you take the weights 
    # (those numbers in the tree leaves that decide predictions), square 'em all, 
    # sum 'em up, and add that as a cost to your loss function. Squaring punishes 
    # big weights more than small ones (unlike L1, which uses absolute values and 
    # can zero 'em out entirely). So, L2 smooths things out gently, shrinking 
    # weights toward zero without killing 'em off, which helps with variance


#### Lesson 3B: XGBoost Intuition (learning rate/ eta, max_depth, num_boost_roung/ n_estimators)

    # 3. Learning Rate/ eta
    # This is like your gas pedal control. It's a number (usually 0.01 to 0.3) that 
    # controls how much weight each tree has on the final prediction.
    # - Low learning rate: Increases both training and inference time. Because it 
    #   increases the number of boosting rounds (trees) - bigger the ensemble, 
    #   slower the eval. But, results in a steady accurate model. 
    # - High learning rate: Speeds things up, but watch out - you could overshoot 
    #   and end up with a bouncy, unstable model. 

    # 4. max_depth 
    # This decides how tall your decision trees can grow - think of it as the 
    # number of questions each tree can ask before it stops splitting hairs. In 
    # tree terms, it's the maximum levels or branches it can have.
    # - Shallow trees (low max_depth, say 3-6): These are simpletons - quick to
    #   train, less likely to overfit because they don't dig into every tiny 
    #   detail. But they might miss deeper patterns, like a kid's drawing versus 
    #   a masterpiece.
    # - Deep trees (high max_depth, like 10+): These bad boys can capture fancy, 
    #   complex relationships in your data. Problem? They love overfitting -
    #   memorizing noise instead of real signals, especially on noisy datasets. 
    #   Pair 'em with regularization to keep 'em in check, or your model turns 
    #   into a fragile mess that crashes on new data.

    # 5. num_boost_round (or n_estimators) 
    # This is just how many trees you slap into your boosting team - the number of 
    # boosting iterations. You can think of it as a 'dance partner' of learning
    # rate/ eta.
    # - Low learning rate/eta -> requires higher n_estimators (500+)
    # - High learning rate/eta -> requires low n_estimators (50-100) 

#### Lesson 3C: XGBoost Intuition (lambda, subsample, colsample_bytree)

    # 6. lambda (L2 regularization)
    # Lambda is XGBoost's knob for L2 regularization. Adds a penalty for big, 
    # flashy coefficients in the leaves. 
    # - Low lambda (close to 0): Lets trees go wild with big swings in predictions. 
    #   Fine if your data's clean, but on noisy crap? Expect overfitting - model's 
    #   chasing ghosts instead of real patterns. 
    # - High lambda (say 1 or more): Cranks up the penalty, shrinking weights and 
    #   smoothing things out. Trees stay modest, reducing variance and fighting 
    #   overfitting. But crank it too high, and you underfit - model becomes a lazy 
    #   slob ignoring useful signals. Balance it. Pro tip: Start at 1 and tune; 
    #   it's like optimizing compiler flags - small tweaks, big wins.

    # 7. subsample
    # This is your data diet plan - subsample decides what fraction of your 
    # training data (rows) each tree gets to munch on. Usually between 0.5 and 
    # 1.0, it's like randomly sampling without replacement for each boosting round.
    # - Low subsample (e.g., 0.5-0.8): Grabs a chunk of data per tree, injecting 
    #   randomness to prevent overfitting. Makes the ensemble tougher and less 
    #   biased to outliers. 
    # - High subsample (close to 1.0): Uses almost all data each time - more 
    #   accurate per tree, but risks memorizing noise if your dataset's messy. Use 
    #   it when data's scarce or clean; otherwise, dial it down to keep things 
    #   general. Remember, variety's the spice - don't let your model gorge on the 
    #   same buffet every round.

    # 8. colsample_bytree
    # Now we're talking features - colsample_bytree picks a random subset of 
    # columns (features) for each tree, usually 0.5 to 1.0. It's like forcing your 
    # team to specialize instead of hogging all the tools.
    # - Low colsample (0.5-0.8): Randomly selects fewer features per tree, adding 
    #   diversity and curbing overfitting. Great for high-dimensional data where 
    #   features correlate - prevents trees from fixating on the same predictors. 
    #   Plus, it trains faster. 
    # - High colsample (close to 1.0): Lets trees see most or all features - 
    #   captures more interactions but invites overfitting if features are 
    #   redundant. Use sparingly unless your data's sparse and pure. Tune this with
    #   subsample for that sweet spot. 

    # Bottom line: These params are your anti-bullshit shields. 
    # - lambda penalizes complexity, 
    # - subsample and colsample inject randomness 
    # Together, they make XGBoost robust without the drama of vanilla boosting.
