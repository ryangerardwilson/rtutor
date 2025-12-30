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

#### Lesson 2: Top 11 Things to Inspect the First Time You Access a Dataframe 

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
    # Filter rows and columns
    row_condition = df.assigned_col.notna()
    df[['mobile', 'account_id', 'assigned', 'otp']][row_condition]
    corr_matrix = df.corr(numeric_only=True)  

    # 2. Duplicate rows & subset
    df.duplicated().sum()
    df.duplicated(subset=['id', 'date']).sum()

    # 3. Missing values
    df.isnull().sum()
    df.isnull().mean() * 100  # % missing
    df = df[df.datetime_col.notna()] # Filter out rows with certain missing
	values

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

    # 6. Value counts
    df.col.value_counts() # Chain: .sort_index(), .round(n), nlargest(n), nsmallest(n)
    df[['categorical_col_1', 'categorical_col_2']].value_counts()
    df.groupby('col').size() # same logic as df.value_counts(), both return Series
    df.groupby(['col1','col2']).size() # works with a list of columns
	
    # 7. Summary stats - look for impossible values (e.g., negative age),
    # extreme outliers, or unexpected categories. Gives: count, unique, mean, freq, 
    # top (mode), std, min, max, quantiles
    df.describe(include='all')
    df.describe(include='all').loc['count'].T # deep dive aesthetically

    # 8. Quantile Analysis
    cut_off = df.probs.quantile(0.90)      
    df['meets_cutoff'] = np.where(df.probs > cut_off,1,0)
    print(df.meets_cutoff.value_counts())

    # 9. Quantile Distribution Analysis 
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

#### Lesson 1: Data Visualization Fundamentals 

    # 1. Normalised vs. De-normalised data

	# Normalization (in the original Codd/1970 relational database sense) =
	# organizing tables to eliminate redundancy and update anomalies by
	# splitting data into many narrow, tightly linked tables following strict
	# normal forms
    #!
	# Core Idea: Every piece of information appears exactly once, and everything else
	# references it via keys.
    #!
	#! Real-world example - fully normalized version of the same data:
	#! Table                Key 	   	 	   Columns
	#! customers            customer_id	   customer_id, name, birth_date, city
	#! orders               order_id	   order_id, customer_id, order_date, amount
	#! order_items          item_id		   item_id, order_id, product_id, qty, price
	#! products             product_id	   product_id, product_name, category
	#! payments             payment_id	   payment_id, order_id, payment_date, method
    #!
	# Why Normalization Is the Enemy of Real-World ML? 
    # 1. To train a model you would have to execute massive star-schema joins + window 
    #    aggregations on-the-fly for every single prediction or training row. That's 
    #    100-1000x slower and leaks like crazy if you're not extremely careful.
    # 2. Normalisation often requrires usage of id attributes, whare are, in
    #    essence, just identifiers - and feeding an ML model an id column will
    #    cause it to learn nothing.

    # 2. Mathematical Data Structures

	# Generally speaking, data appears in the real world in the following
	# mathematical forms:
	# a. Event/ logs: they simply record a series of events around a time axis
	# b. Real world Tabular Data: a denormalized, point-in-time feature matrix. Note
	#    that this is not a relation, but a matrix, which is a superset of a relation.
	# c. Codd's original 1970 relational model - a time-varying relation that holds
	#    only currently valid tuples
    #!
	# What is a good data set to model?
	# A good dataset for supervised learning is a point-in-time feature table, rooted 
    # in the REAL WORLD, meeting the following conditions:
	# - Each row corresponds to a unique real world entity (e.g., customer, vehicle, 
    #   patient, etc.) observed at a specific prediction timestamp t_pred, collectively, 
    #   creating a 'Monte-Carlo' sample (Whats a Monte-Carlo sample? Let reality roll 
    #   the dice for you, then write down what actually happened. You repeat that many 
    #   times, and you get a bunch of real-world random samples.
	# - The target variable y_i is the outcome of interest that materializes strictly 
    #   after t_pred (e.g., failure within next 30 days, churn in the following month, 
    #   etc.)
	# - For every feature x_i, the value j in the row must be known or computable using 
    #   only information available at or before t_pred. No feature may incorporate data 
    #   from any time > t_pred (this prohibition is called 'future leakage' or 'target 
    #   leakage')
	# - There are no duplicate rows that are invalid Monte-Carlo samples
    # - There is no id column as a feature - else, the model will learn nothing.
    #! 	
	# Why do 'good' duplicates (i.e. duplicates that are valid Monte-Carlo samples) 
    # actually add value?
	# - They teach the model the actual odds: when everything looks exactly like this, 
    #   it fails x% of the time.
	# - If you delete all of them and keep only one, the model thinks 'this never 
    #   happens' or 'this always happens', causing totally wrong probabilities.
	# - Every extra copy is like flipping the real-world coin one more time and writing 
    #   down what actually happened.

    # 3. Using Definitional Precision to Prevent Target Leakage

	# Now, the following definitions become clear:
	# Feature: Anything known ON_OR_BEFORE prediction_time
	# Target: A value that is realized AFTER prediction_time. For instance,
	# - for linear regression: number OR log(number) AFTER {prediction_time}
	# - for binary classification: 1 IF {condition} AFTER {prediction_time} ELSE 0
    #!
	# To understand the importance of the above definitions, lets say we define our
	# target as follows: WITH {prediction_time}: 1 IF {condition} ELSE 0
	# The problem that is created is called 'target leakage', where the rule
	# governing the impact of prediction_time on the outcome is ambiguously
	# stated.
    #!
	# Thus, in preprocessing raw data, it is imperative to ensure that the data
	# filtered such that:
	# - target is AFTER prediction_time
	# - all features exist ON_OR_BEFORE prediction_time

#### Lesson 2: Feature Engineering Checklist

    # 1. prediction_time, num_features, cat_features, and target should be defined 
    from datetime import datetime
    prediction_time = datetime(2025, 12, 29, 14, 30, 0)  
    print(prediction_time)  
    #! 2025-12-29 14:30:00
    event_timestamp = 'date' # This is the name of the col used to do a
                             # test-train split qua the prediction_time
    num_features = []
    cat_features = []
    event_timestamp = []
    target = 'target_col'
    
    # 2. The set of rows should have no duplicates and constitue a 'monte carlo' sample
    df = df.drop_duplicates()
    
    # 3. Column names should have consistent lower case formatting
    # df.columns = df.columns.str.lower()

    # 4. The target column must have a numeric dtype - either 0/1 for binary
    # classification, or a number like price, category code, etc. 
    # - If the target column is categorical in nature, assign a number to each 
    #   category, as is shown below.
    target_number_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}
    df[target] = df[target].map(target_number_dict)
    # - If the target column is boolean type (for binary classification),
    #   do as below.
    df[target] = df[target].astype(int)

    # 5. All numeric features must have pandas numeric dtype, and all categorical
    # features must have category dtype. 
    for col in df.columns.to_list():
        try:
            if col in num_features:
                df[col] = pd.to_numeric(df[col])
            elif col in cat_features:
                df[col] = df[col].astype('category')
            else:
                pass
        except Exception as e:
            print(f'Needs more work: {col}')
            print(e)

    # 6. The 'no-category' value in categorical features should be consistently be 
    # 'unknown'
    df['credit_mix'] = df['credit_mix'].cat.add_categories('unknown')   
    df['credit_mix'] = df['credit_mix'].fillna('unknown')         
    df.loc[df['credit_mix'] == '_', 'credit_mix'] = 'unknown'  
    df['credit_mix'] = df['credit_mix'].cat.remove_unused_categories()  

    # 7. All categorical feautures should have >2 categories, else they should
    # transformed as 0/1 numeric feautres
    for col in cat_features:
        n_unique = df[col].nunique()
        unique_vals = non_null.unique().tolist()
        if n_unique <= 2:
            print(f'Action required for column: {col}')
    
    # 8. An id column should never be a feature, unless it is more or less a
    # categorical feature 
    num_features.remove('id')

    # 9. Do NOT eliminate features at this stage, becuase the simplest
    # test of target leakage/ weak features is the best_features_df which we
    # will create after training the model. Once, we have that, we can apply
    # the following principles: 
    # - If one or a few features dominate the importance scores (e.g., one feature 
    #   with 0.5+ normalized gain while others are near zero), it could signal leakage. 
    #   This happens because leaky features often provide 'shortcut' information that's 
    #   too directly tied to the target, making them overly influential in tree splits.
    # - Features with near-zero normalized gain (e.g., below 0.01 i.e. 1%, contribute 
    #   little and can often be dropped without harming performance. You can do
    #   this manually, or by using recursive feature elimination.

    # 10. Transfer the data from the Feature Engineering API to the Model API
    df.to_parquet('step1.parquet')
    with open('step1_lists.pkl', 'wb') as f:
        pickle.dump({
            'event_timestamp': event_timestamp,
            'num_features': num_features,
            'cat_features': cat_features,
            'target': target
        }, f)

    # Access the data from the Model API
    df = pd.read_parquet('step1.parquet')
    with open('step1_lists.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
        locals().update(loaded_data)

#### Lesson 3: Decision Trees

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

#### Lesson 4: XGBoost Intuition

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

#### Lesson 5: Binary Classification Intuition

    # 1. Metrics
    #!----------

    # 1.1. Base Rate
    # Base rate is simply the mean of the target column of the test data. It's the 
    # benchmark for 'doing nothing smart' - if you randomly selected users to 
    # target, your expected conversion rate would be exactly this base rate. It is 
    # important when we calculate lift (lift = precision/ base_rate).

    # 1.2. AUC
    # Imagine you randomly pick:
    # - one user who actually converted (positive class)
    # - one user who did not convert (negative class)
    # 
    # The AUC is exactly the probability that your model assigns a 'higher 
    # predicted probability' to the 'positive class'/ 'actual converter' than to 
    # the 'negative class'/'non-converter'.
    # - AUC = 1.0 -> Perfect model: it always ranks real converters higher than 
    #   non-converters.
    # - AUC = 0.5 -> Model is no better than random guessing (like flipping a coin)
    # - AUC < 0.5 -> Worse than random (the model is systematically wrong — you 
    #   could just invert its predictions).
    # 
    # In XGBoost lingo, when we say that a model 'ranks well', we mean that it
    # 'ranks well above' i.e.
    # - You sort all instances by the model's output score (descending: highest 
    #   probability first).
    # - A 'good ranking' model places as many true positives as possible above 
    #   (i.e., with higher scores than) true negatives. 
    #   - If the model does this well (auc > 0.7), the top of the list (e.g., your 
    #     P95/ P90) will be enriched with positives - high precision, high lift, 
    #     good recall.
    #   - If it does poorly (auc < 0.6), positives and negatives are likely 
    #     intermixed throughout the list, leading to low enrichment even at the 
    #     top.
    #   - between 0.6 and 0.7, AUC may not conclusively indicate the model's 
    #     performance at p90 and p95. In such cases, we need to see if the
    #     is nevertheless useful for specific real world applications.

    # 1.3. precision
    # Of the users we predict will convert (i.e., we target/select them), what 
    # fraction actually convert? If precision = 0.80 at top 5%, that means 80% 
    # of the users we target actually convert -> very efficient campaign.
    # Formula: precision = true_positives / (true_positives + false_positives)
    # Intuition: High precision -> When we say 'this user will convert', we're 
    # usually right. This metric is critical when false positives are expensive 

    # 1.4. recall (also called sensitivity or true_positive_rate)
    # Of all the users who actually convert, what fraction did we correctly identify?
    # Formula: Recall = true_positives / (true_positives + false_negatives)
    # Intuition: High recall -> We capture most of the real converters. Important 
    # when missing identifying a positive outcome/ missing a converter is costly 
    # (e.g., losing a high-value customer).

    # 1.5. f1_score
    # The harmonic mean of precision and recall - a single score that balances both.
    # Formula: F1 = 2 × (precision × recall) / (precision + recall)

    # 1.6. Accuracy
    # Overall, what fraction of predictions (both convert and not convert) are correct?
    # Formula: accuracy = (true_positives + true_negatives) / total_users
    # Intuition: Simple and intuitive at first glance.
    # Business takeaway: Almost never use accuracy as the primary metric in conversion 
    # prediction. Always prefer AUC, precision, recall, or lift.

    # 1.7. Lift
    # How much better does our targeted group convert compared to the 
    # average (base rate)?
    # Formula: Lift = precision / base_rate
    # Intuition: The most business-friendly metric. Answers: 'If I target these users, 
    # how many times better do they perform than random selection?'

    # 2. Real World Impact Test
    #!-------------------------

    # 1. Do we have a 'valid' model?
    # - if AUC >= 0.8: yes 
    # - else if AUC >= 0.6: yes (implies features set is noisy/imbalanced)
    # - else AUC < 0.6: hard no. Too weak—back to the drawing board.

    # 2. Does the model address the business problem?
    # In the vast majority of real-world binary classification applications—such as 
    # user targeting, conversion prediction, churn prevention, lead scoring, fraud 
    # alerting, or credit risk-the primary deployment mode is ranking and selective 
    # action, not automated hard yes/no decisions across the entire population. This 
    # means we rarely classify every instance with a fixed threshold (e.g., 0.5) or 
    # aim for balanced precision/recall globally. Instead, we use the model’s scores 
    # to rank instances and act only on a constrained subset (e.g., users above 
    # P60-P95 by score), where budget, capacity, or cost constraints apply.
    # The key business decision therefore revolves around the efficiency vs. coverage 
    # trade-off when selecting that actionable subset:
    # - Efficiency (driven by precision at high percentiles, e.g., P95–P99). Measures 
    #   how accurate your positive predictions are when you act on the highest-scored 
    #   instances. High efficiency minimizes wasted resources on false positives. 
    #   This is critical when actions are expensive (e.g., personalized incentives, 
    #   manual reviews, high-value offers). Even models with modest overall AUC 
    #   (0.60-0.70) can deliver strong efficiency if the P95-P99 range shows good lift 
    #   and high precision.
    # - Coverage (driven by recall at lower percentiles, e.g., P60–P90). Measures how 
    #   many of the true positives you capture when casting a somewhat wider net (i.e., 
    #   lowering the score threshold). High coverage reduces missed opportunities 
    #   (false negatives). This matters when missing a positive is costly (e.g., losing 
    #   high-value churners, failing to retain convertible users, or overlooking growth 
    #   opportunities).
    # In practice, these are in direct tension: tightening the threshold for higher 
    # efficiency (acting only above P95-P99) almost always reduces coverage, and vice 
    # versa (acting down to P70-P80 increases coverage at the cost of efficiency). The 
    # F1 score can help when you truly need balance, but most production scenarios lean 
    # toward one side based on constraints.The actionables here, can be sumamrized as 
    # follows (and depends on whether or not False positives would cost the
    # business - for example an expensive discount voucher):
    #!-------------------------------------------------------------------------------------
    #!       goal | fp_expensive | metric_focus |                               threshold |
    #!-------------------------------------------------------------------------------------
    #! efficiency |         True |    above P95 |   precision ≥50%, lift >4x, recall ≥30% |
    #! efficiency |        False |    above P95 | precision ≥40%, lift >2.5x, recall ≥15% |
    #!   coverage |         True |    above P90 |   recall ≥50%, precision ≥40%, lift >3x |
    #!   coverage |        False |   above Pxx* |   recall ≥40%, precision ≥30%, lift >2x |
    #!-------------------------------------------------------------------------------------
    #! *Pxx denotes any P95,P80... etc., as long as the treshold is met

#### Lesson 6: Binary Classification Implementation 

    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )
    from sklearn.feature_selection import RFE

    #! features = ['feat_0', 'feat_1', ... 'feat_19']
    #! tabular_data_df.head(5)
    #!            feat_0    feat_1    feat_2   ...   feat_19     target
    #! user_id                                 ...
    #! 0        0.025729  0.165038  0.072194   ...  0.018873          0
    #! 1        0.054640  0.008674  0.019949   ...  0.033492          0
    #! 2        0.006200  0.032563  0.001667   ...  0.018747          0
    #! 3        0.026806  0.017243  0.096114   ...  0.006708          0
    #! 4        0.120043  0.058937  0.024257   ...  0.006892          0

    class AUCMaximizer:
        def __init__(self, tabular_data_df, features, target):
            self.tabular_data_df = tabular_data_df
            self.features = features
            self.target = target
            self.n_features_to_select = 10
            self.n_trials = 30
            self.test_size = 0.2
            self.val_size = 0.2
            self.random_state = 42
            self.default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
            self.num_boost_round = 200
            self.early_stopping_rounds = 20
            self.optuna_boost_round = 1000
            self.optuna_early_stopping = 20
            self.optuna_search_spaces = {
                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                'eta': {'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
            self.results = {}
            self.comparative_df = None
            self.best_name = None
            self.best_auc = None
            self.model = None
            self.X_test_selected = None
            self.y_test = None
            self.selected_features = None
            self.y_pred_test = None
            self.base_rate = None
            self.best_features_df = None

        def manual_without_rfe(self):
            X_train, X_test, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            
            selected_features = self.features  # No RFE, use all features
            
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            
            params = self.default_params.copy()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            return model, X_test, y_test, selected_features

        def manual_with_rfe(self):
            X_train, X_test, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            
            # RFE for feature selection
            base_model = xgb.XGBClassifier(
                **self.default_params,
                random_state=self.random_state,
                enable_categorical=True,
            )
            rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            print("\n=== Selected Features from RFE (Manual) ===")
            print(selected_features)
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
            
            params = self.default_params.copy()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            return model, X_test_selected, y_test, selected_features

        def automated_without_rfe(self):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                self.tabular_data_df.drop(self.target, axis=1),
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y_train_full
            )
            
            selected_features = X_train.columns.tolist()  # No RFE, use all features
            
            # Define the objective function for Optuna
            def objective(trial):
                max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
                eta = trial.suggest_float('eta', 
                                          self.optuna_search_spaces['eta']['low'], 
                                          self.optuna_search_spaces['eta']['high'], 
                                          log=self.optuna_search_spaces['eta']['log'])
                subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
                colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                }
                dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.optuna_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.optuna_early_stopping,
                    verbose_eval=False,
                )
                y_pred_val = model.predict(dval)
                auc = roc_auc_score(y_val, y_pred_val)
                return auc
            
            # Create and optimize the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'auc'
            
            print('Best hyperparameters found by Optuna (without RFE):')
            print(best_params)
            
            # Train the final model with best params on full train set
            dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            
            model = xgb.train(
                best_params,
                dtrain_full,
                num_boost_round=self.optuna_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            return model, X_test, y_test, selected_features

        def automated_with_rfe(self):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                self.tabular_data_df.drop(self.target, axis=1),
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y_train_full
            )
            
            # RFE for feature selection
            base_model = xgb.XGBClassifier(
                **self.default_params,
                random_state=self.random_state,
                enable_categorical=True,
            )
            rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            print("\n=== Selected Features from RFE (Automated) ===")
            print(selected_features)
            
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            X_train_full_selected = X_train_full[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Define the objective function for Optuna on selected features
            def objective(trial):
                max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
                eta = trial.suggest_float('eta', 
                                          self.optuna_search_spaces['eta']['low'], 
                                          self.optuna_search_spaces['eta']['high'], 
                                          log=self.optuna_search_spaces['eta']['log'])
                subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
                colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                }
                dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val_selected, label=y_val, enable_categorical=True)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.optuna_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.optuna_early_stopping,
                    verbose_eval=False,
                )
                y_pred_val = model.predict(dval)
                auc = roc_auc_score(y_val, y_pred_val)
                return auc
            
            # Create and optimize the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'auc'
            
            print('Best hyperparameters found by Optuna (with RFE):')
            print(best_params)
            
            # Train the final model with best params on full train set with selected features
            dtrain_full = xgb.DMatrix(X_train_full_selected, label=y_train_full, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
            
            model = xgb.train(
                best_params,
                dtrain_full,
                num_boost_round=self.optuna_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            return model, X_test_selected, y_test, selected_features

        def run_all(self):
            print("\n=== 1. Manual without RFE ===")
            model1, X_test1, y_test, sel1 = self.manual_without_rfe()
            y_pred1 = model1.predict(xgb.DMatrix(X_test1))
            auc1 = roc_auc_score(y_test, y_pred1)
            print(f"AUC: {auc1:.4f}")
            self.results["Manual without RFE"] = (auc1, model1, X_test1, y_test, sel1, y_pred1)

            print("\n=== 2. Manual with RFE ===")
            model2, X_test2, y_test, sel2 = self.manual_with_rfe()
            y_pred2 = model2.predict(xgb.DMatrix(X_test2))
            auc2 = roc_auc_score(y_test, y_pred2)
            print(f"AUC: {auc2:.4f}")
            self.results["Manual with RFE"] = (auc2, model2, X_test2, y_test, sel2, y_pred2)

            print("\n=== 3. Automated (Optuna) without RFE ===")
            model3, X_test3, y_test, sel3 = self.automated_without_rfe()
            y_pred3 = model3.predict(xgb.DMatrix(X_test3))
            auc3 = roc_auc_score(y_test, y_pred3)
            print(f"AUC: {auc3:.4f}")
            self.results["Automated without RFE"] = (auc3, model3, X_test3, y_test, sel3, y_pred3)

            print("\n=== 4. Automated (Optuna) with RFE ===")
            model4, X_test4, y_test, sel4 = self.automated_with_rfe()
            y_pred4 = model4.predict(xgb.DMatrix(X_test4))
            auc4 = roc_auc_score(y_test, y_pred4)
            print(f"AUC: {auc4:.4f}")
            self.results["Automated with RFE"] = (auc4, model4, X_test4, y_test, sel4, y_pred4)

        def build_comparative(self):
            comparative_data = {
                'model': list(self.results.keys()),
                'auc': [self.results[k][0] for k in self.results],
                'num_features': [len(self.results[k][4]) for k in self.results]
            }
            self.comparative_df = pd.DataFrame(comparative_data)
            self.comparative_df = self.comparative_df.sort_values(by='auc', ascending=False).reset_index(drop=True)

        def select_best(self):
            self.best_name = self.comparative_df.iloc[0]['model']
            self.best_auc, self.model, self.X_test_selected, self.y_test, self.selected_features, self.y_pred_test = self.results[self.best_name]
            self.base_rate = self.y_test.mean()

        def optimize(self):
            self.run_all()
            self.build_comparative()
            self.select_best()
            
            # Compute feature importance for the best model
            importance_gain = self.model.get_score(importance_type="gain")
            if importance_gain:
                total_gain = sum(importance_gain.values())
                normalized_gain = {feat: gain / total_gain for feat, gain in importance_gain.items()}
                self.best_features_df = pd.DataFrame({
                    "feature": list(normalized_gain.keys()),
                    "importance_gain_normalized": list(normalized_gain.values()),
                }).sort_values(by="importance_gain_normalized", ascending=False)
                self.best_features_df["importance_rank"] = range(1, len(self.best_features_df) + 1)
                self.best_features_df = self.best_features_df.set_index("importance_rank")
            else:
                self.best_features_df = pd.DataFrame()
            
            return {
                'comparative_df': self.comparative_df,
                'best_name': self.best_name,
                'best_auc': self.best_auc,
                'model': self.model,
                'X_test_selected': self.X_test_selected,
                'y_test': self.y_test,
                'selected_features': self.selected_features,
                'y_pred_test': self.y_pred_test,
                'base_rate': self.base_rate,
                'best_features_df': self.best_features_df
            }

        def print_results(self):
            print("\n=== Comparative Model Results ===")
            print(self.comparative_df.to_string(index=False))

            print("\n" + "="*50)
            print(f"BEST MODEL: {self.best_name}")
            print(f"Best Test AUC: {self.best_auc:.4f}")
            print("="*50)

            print("\nSelected Features:")
            print(self.selected_features)

            print(f"\nBase conversion rate on test set: {self.base_rate:.4f}")

            print(f'AUC on test set (best model): {self.best_auc:.4f}\n')

    class MetricsComputer:
        def __init__(self, y_test, y_pred_test, base_rate=None):
            self.y_test = y_test
            self.y_pred_test = y_pred_test
            self.base_rate = base_rate if base_rate is not None else y_test.mean()

        def compute_metrics(self):
            percentiles = [99] + list(range(95, 0, -5)) + [1]
            table_rows = []
            for p in percentiles:
                cutoff = np.percentile(self.y_pred_test, p)
                y_pred_binary = (self.y_pred_test >= cutoff).astype(int)
                tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_binary).ravel()
                precision = precision_score(self.y_test, y_pred_binary, zero_division=0)
                recall = recall_score(self.y_test, y_pred_binary, zero_division=0)
                f1 = f1_score(self.y_test, y_pred_binary, zero_division=0)
                accuracy = accuracy_score(self.y_test, y_pred_binary)
                lift = precision / self.base_rate if self.base_rate > 0 and precision > 0 else 0
                table_rows.append({
                    'percentile': f'P{p}',
                    'cutoff_prob': round(cutoff, 4),
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1': round(f1, 4),
                    'accuracy': round(accuracy, 4),
                    'lift': round(lift, 2),
                })
            metrics_df = pd.DataFrame(table_rows).set_index('percentile')
            return metrics_df

    # Example usage
    maximizer = AUCMaximizer(tabular_data_df, features, 'converted')
    results = maximizer.optimize()
    maximizer.print_results()

    print("\n=== best_features_df (Top Features by Gain) ===")
    print(results['best_features_df'].to_string(float_format="{:.4f}".format))

    metrics_comp = MetricsComputer(results['y_test'], results['y_pred_test'], results['base_rate'])
    metrics_df = metrics_comp.compute_metrics()
    print("\n=== Performance by Percentile Threshold (Best Model) ===")
    print(metrics_df.to_string())
    #!             cutoff_prob   tp    fp   fn    tn  precision  recall      f1  accuracy  lift
    #! percentile
    #! P1               0.3478    7    13  253  1727     0.3500  0.0269  0.0500     0.867  2.69
    #! P5               0.2452   27    73  233  1667     0.2700  0.1038  0.1500     0.847  2.08
    #! ...
    #! P70              0.0879  207  1193   53   547     0.1479  0.7962  0.2494     0.377  1.14
    #! P75              0.0829  215  1285   45   455     0.1433  0.8269  0.2443     0.335  1.10
    #! P80              0.0776  223  1377   37   363     0.1394  0.8577  0.2398     0.293  1.07
    #! P85              0.0725  232  1468   28   272     0.1365  0.8923  0.2367     0.252  1.05
    #! P90              0.0663  243  1557   17   183     0.1350  0.9346  0.2359     0.213  1.04
    #! P95              0.0587  251  1649    9    91     0.1321  0.9654  0.2324     0.171  1.02
    #! P99              0.0493  258  1722    2    18     0.1303  0.9923  0.2304     0.138  1.00

    print("\nNOTE:")
    print("- 'Pxx' means selecting users with predicted probability >= xx-th percentile")
    print("- Higher percentile = stricter threshold = higher precision, lower recall")
    print("- Lift = precision / base_rate")

    # Demonstrate making a prediction
    example_row = tabular_data_df.iloc[0][results['selected_features']]
    dexample = xgb.DMatrix(pd.DataFrame([example_row]))
    example_pred = results['model'].predict(dexample)[0]

    print(f"\n=== Example Prediction ===")
    print(f"Input features:\n{results['selected_features']}")
    print(f"Predicted probability: {example_pred:.4f}")

#### Lesson 7: Linear Regression Intuition

    # 1. Metrics
    #!----------

    # 1.1. Baseline Error
    # The baseline error is the error you'd get by always predicting the mean of 
    # the target variable in the test data. It's the benchmark for 'doing nothing 
    # smart'-if you ignored all features and just used the average value as your 
    # prediction for every instance, this would be your error level. It's crucial 
    # for calculating relative improvements (e.g., how much better your model is 
    # than this naive approach). In regression, the total variance of the target 
    # is like the 'base rate' equivalent, and metrics like R-squared compare your 
    # model's error to this baseline.

    # 1.2. Mean Squared Error (MSE)
    # MSE measures the average squared difference between your model's predictions 
    # and the actual values. It penalizes larger errors more heavily because of 
    # the squaring.
    # Formula: 
    mse = sum((actual - predicted)**2 for actual, predicted in zip(actuals, predicteds)) / len(actuals)
    # Intuition: Low MSE means your predictions are close to the true values on 
    # average, with big mistakes being rare. It's sensitive to outliers (a single 
    # large error can spike it). Use MSE when you care more about avoiding big 
    # prediction misses, like in financial forecasting where over/underestimating 
    # by a lot is costly.

    # 1.3. Root Mean Squared Error (RMSE)
    # RMSE is the square root of MSE, bringing the error back to the original units 
    # of the target variable (e.g., if predicting house prices in dollars, RMSE is 
    # in dollars).
    # Formula: 
    rmse = math.sqrt(mse)
    # Intuition: Easier to interpret than MSE-e.g., RMSE = 10 means your predictions 
    # are off by about 10 units on average. Like MSE, it's outlier-sensitive. 
    # Critical when you need an intuitive sense of error magnitude, such as 'our 
    # sales predictions are typically off by $500.'

    # 1.4. Mean Absolute Error (MAE)
    # MAE measures the average absolute difference between predictions and actuals, 
    # without squaring-so it treats all errors equally.
    # Formula: 
    mae = sum(abs(actual - predicted) for actual, predicted in zip(actuals, predicteds)) / len(actuals)
    # Intuition: Robust to outliers (a huge error doesn't dominate as much). High MAE 
    # means frequent small-to-medium errors. Use when all errors matter similarly, 
    # like in inventory prediction where being off by 1 unit costs the same regardless of scale.

    # 1.5. R-squared (Coefficient of Determination)
    # R-squared tells you what fraction of the target's variance is explained by 
    # your model.
    # Formula: 
    mse_baseline = sum((actual - mean_actual)**2 for actual in actuals) / len(actuals)
    r_squared = 1 - (mse_model / mse_baseline) 
    # Intuition:
    # - R_squared = 1.0 -> Perfect model: explains all variance, no error left.
    # - R_squared = 0.0 -> Model is no better than predicting the mean (random baseline).
    # - R_squared < 0.0 -> Worse than baseline (model adds noise—fix it!).
    # Busisness takeaway: R_squared > 0.7 often means a strong model for many applications, 
    # but context matters (e.g., in noisy social sciences, >0.3 might be decent).

    # 1.6. Adjusted R-squared
    # Adjusted AR_squared penalizes adding irrelevant features to the model, unlike plain 
    # R_squared which can increase just by adding more variables.
    # Formula: where n = number of samples, k = number of predictors
    AR_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))  
    # Intuition: Use this when comparing models with different numbers of features-prevents 
    # overfitting illusion. If AR_squared >> R_squared, your model might have too many useless 
    # variables.

    # 1.7. Mean Absolute Percentage Error (MAPE)
    # MAPE measures the average percentage error between predictions and actuals.
    # Formula: 
    mape = (100 / len(actuals)) * sum(abs((actual - predicted) / actual) for actual, predicted in zip(actuals, predicteds) if actual != 0)
    # Intuition: Great for relative errors, e.g., 'predictions are off by 5% on average.' Avoid 
    # if actuals can be zero (division by zero). Useful in sales or demand forecasting where 
    # percentage accuracy matters.

    # 2. Real World Impact Test
    #!-------------------------

    # 1. Do we have a 'valid' model?
    # - if R_squared >= 0.5: yes 
    # - else if R_squared >= 0.3: yes (implies features set is noisy/imbalanced)
    # - else R_squared < 0.3: hard no. Too weak—back to the drawing board.

    # 2. Does the model address the business problem?
    # In most real-world regression applications-like sales forecasting, price prediction, 
    # risk scoring, or demand estimation—the goal is accurate point estimates within 
    # constraints, not perfect fits. We deploy models to guide decisions under uncertainty, 
    # balancing accuracy, interpretability, and cost.
    # Key trade-offs:
    # - Accuracy (driven by low RMSE/MAE): How close are predictions to reality? Critical 
    #   when errors are costly (e.g., overstocking inventory = waste; understocking = lost sales).
    # - Interpretability: Can we understand feature impacts (coefficients)? Vital for 
    #   trust and compliance (e.g., why does age affect credit score?).
    # - Robustness: Performs well on new data? Check for overfitting via cross-validation.
    # In practice, focus on error distribution: e.g., median error for typical cases, worst-case 
    # for risks.
    # The actionables can be summarized as follows:
    #!----------------------------------------------------------------------------------
    #!        goal | metric_focus |         does_it_address_business_problem_threshold |
    #!----------------------------------------------------------------------------------
    #!    accuracy |     RMSE/MAE |         RMSE < 10% of mean target, R_squared > 0.6 |
    #!  robustness | CV R_squared | CV R_squared > 0.5, drop <10% from train R_squared |
    #!----------------------------------------------------------------------------------

#### Lesson 8: Supervised Regression Implementation 

    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import optuna
    from scipy.stats import skew
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        root_mean_squared_error,
    )
    from sklearn.feature_selection import RFE

    #! features = ['feat_0', 'feat_1', ... 'feat_19']
    #! print(tabular_data_df.head(10))
    #!       feat_0    feat_1    feat_2  ...   feat_18   feat_19    target
    #! id                                ...
    #! 0   0.496714 -0.138264  0.647689  ... -0.908024 -1.412304  7.100724
    #! 1   1.465649 -0.225776  0.067528  ... -1.328186  0.196861  4.862268
    #! 2   0.738467  0.171368 -0.115648  ...  0.331263  0.975545  2.674120
    #! 3  -0.479174 -0.185659 -1.106335  ...  0.091761 -1.987569  1.147614
    #! 4  -0.219672  0.357113  1.477894  ...  0.005113 -0.234587  2.540443
    #! 5  -1.415371 -0.420645 -0.342715  ...  1.142823  0.751933  0.917154
    #! 6   0.791032 -0.909387  1.402794  ...  0.813517 -1.230864  6.398368
    #! 7   0.227460  1.307143 -1.607483  ... -1.191303  0.656554  1.595872
    #! 8  -0.974682  0.787085  1.158596  ... -0.264657  2.720169  1.484465
    #! 9   0.625667 -0.857158 -1.070892  ...  0.058209 -1.142970  3.685111

    # 1. Skewness check and possible target variable transformation
    skew_value = skew(tabular_data_df['target'])
    log_transformation_needed = abs(skew_value) > 0.5
    if log_transformation_needed:
        tabular_data_df['target'] = np.log(tabular_data_df['target'])
    #!       feat_0    feat_1    feat_2  ...   feat_18   feat_19    target
    #! id                                ...
    #! 0   0.496714 -0.138264  0.647689  ... -0.908024 -1.412304  1.960197
    #! 1   1.465649 -0.225776  0.067528  ... -1.328186  0.196861  1.581505
    #! 2   0.738467  0.171368 -0.115648  ...  0.331263  0.975545  0.983620
    #! 3  -0.479174 -0.185659 -1.106335  ...  0.091761 -1.987569  0.137685
    #! 4  -0.219672  0.357113  1.477894  ...  0.005113 -0.234587  0.932338
    #! 5  -1.415371 -0.420645 -0.342715  ...  1.142823  0.751933 -0.086480
    #! 6   0.791032 -0.909387  1.402794  ...  0.813517 -1.230864  1.856043
    #! 7   0.227460  1.307143 -1.607483  ... -1.191303  0.656554  0.467420
    #! 8  -0.974682  0.787085  1.158596  ... -0.264657  2.720169  0.395054
    #! 9   0.625667 -0.857158 -1.070892  ...  0.058209 -1.142970  1.304301

    class R2Maximizer:
        def __init__(self, tabular_data_df, features, target):
            self.tabular_data_df = tabular_data_df
            self.features = features
            self.target = target
            self.n_features_to_select = 10
            self.n_trials = 30
            self.test_size = 0.2
            self.val_size = 0.2
            self.random_state = 42
            self.default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
            self.num_boost_round = 200
            self.early_stopping_rounds = 20
            self.optuna_boost_round = 1000
            self.optuna_early_stopping = 20
            self.optuna_search_spaces = {
                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                'eta': {'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
            self.log_transformation_needed = False
            self.results = {}
            self.comparative_df = None
            self.best_name = None
            self.best_r2 = None
            self.best_mae = None
            self.best_rmse = None
            self.model = None
            self.X_test_selected = None
            self.y_test = None
            self.selected_features = None
            self.y_pred_test = None
            self.y_test_orig = None
            self.y_pred_orig = None
            self.base_mean = None
            self.baseline_r2 = None
            self.baseline_mae = None
            self.baseline_rmse = None
            self.best_features_df = None

            # Skewness Check and log transformation of target
            skew_value = skew(self.tabular_data_df[self.target])
            print(f"\nSkewness of original target: {skew_value:.4f}")
            if abs(skew_value) > 0.5:
                print("Target is skewed. Applying log transformation.")
                self.tabular_data_df[self.target] = np.log(self.tabular_data_df[self.target])
                skew_transformed = skew(self.tabular_data_df[self.target])
                print(f"Skewness after log transformation: {skew_transformed:.4f}")
                self.log_transformation_needed = True
            else:
                print("Target is approximately normal. No transformation applied.")

        def compute_baseline(self):
            X_train, _, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
            )
            y_train_orig = np.exp(y_train) if self.log_transformation_needed else y_train
            y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
            mean_target = y_train_orig.mean()
            baseline_pred = np.full_like(y_test_orig, mean_target)
            r2 = r2_score(y_test_orig, baseline_pred)
            mae = mean_absolute_error(y_test_orig, baseline_pred)
            rmse = root_mean_squared_error(y_test_orig, baseline_pred)
            return r2, mae, rmse, mean_target, y_test_orig, baseline_pred

        def manual_without_rfe(self):
            X_train, X_test, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
            )
            
            selected_features = self.features  # No RFE, use all features
            
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            
            params = self.default_params.copy()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            return model, X_test, y_test, selected_features

        def manual_with_rfe(self):
            X_train, X_test, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
            )
            
            # RFE for feature selection
            base_model = xgb.XGBRegressor(
                **self.default_params,
                random_state=self.random_state,
                enable_categorical=True,
            )
            rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            print("\n=== Selected Features from RFE (Manual) ===")
            print(selected_features)
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
            
            params = self.default_params.copy()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            return model, X_test_selected, y_test, selected_features

        def automated_without_rfe(self):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                self.tabular_data_df.drop(self.target, axis=1),
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
            )
            
            selected_features = X_train.columns.tolist()  # No RFE, use all features
            
            # Define the objective function for Optuna
            def objective(trial):
                max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
                eta = trial.suggest_float('eta', 
                                          self.optuna_search_spaces['eta']['low'], 
                                          self.optuna_search_spaces['eta']['high'], 
                                          log=self.optuna_search_spaces['eta']['log'])
                subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
                colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                }
                dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.optuna_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.optuna_early_stopping,
                    verbose_eval=False,
                )
                y_pred_val = model.predict(dval)
                y_val_orig = np.exp(y_val) if self.log_transformation_needed else y_val
                y_pred_orig = np.exp(y_pred_val) if self.log_transformation_needed else y_pred_val
                r2 = r2_score(y_val_orig, y_pred_orig)
                return r2
            
            # Create and optimize the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_params['objective'] = 'reg:squarederror'
            best_params['eval_metric'] = 'rmse'
            
            print('Best hyperparameters found by Optuna (without RFE):')
            print(best_params)
            
            # Train the final model with best params on full train set
            dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            
            model = xgb.train(
                best_params,
                dtrain_full,
                num_boost_round=self.optuna_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            return model, X_test, y_test, selected_features

        def automated_with_rfe(self):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                self.tabular_data_df.drop(self.target, axis=1),
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
            )
            
            # RFE for feature selection
            base_model = xgb.XGBRegressor(
                **self.default_params,
                random_state=self.random_state,
                enable_categorical=True,
            )
            rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            print("\n=== Selected Features from RFE (Automated) ===")
            print(selected_features)
            
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            X_train_full_selected = X_train_full[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Define the objective function for Optuna on selected features
            def objective(trial):
                max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
                eta = trial.suggest_float('eta', 
                                          self.optuna_search_spaces['eta']['low'], 
                                          self.optuna_search_spaces['eta']['high'], 
                                          log=self.optuna_search_spaces['eta']['log'])
                subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
                colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                }
                dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val_selected, label=y_val, enable_categorical=True)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.optuna_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.optuna_early_stopping,
                    verbose_eval=False,
                )
                y_pred_val = model.predict(dval)
                y_val_orig = np.exp(y_val) if self.log_transformation_needed else y_val
                y_pred_orig = np.exp(y_pred_val) if self.log_transformation_needed else y_pred_val
                r2 = r2_score(y_val_orig, y_pred_orig)
                return r2
            
            # Create and optimize the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_params['objective'] = 'reg:squarederror'
            best_params['eval_metric'] = 'rmse'
            
            print('Best hyperparameters found by Optuna (with RFE):')
            print(best_params)
            
            # Train the final model with best params on full train set with selected features
            dtrain_full = xgb.DMatrix(X_train_full_selected, label=y_train_full, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
            
            model = xgb.train(
                best_params,
                dtrain_full,
                num_boost_round=self.optuna_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            return model, X_test_selected, y_test, selected_features

        def run_all(self):
            self.baseline_r2, self.baseline_mae, self.baseline_rmse, self.base_mean, _, _ = self.compute_baseline()

            print("\n=== 1. Manual without RFE ===")
            model1, X_test1, y_test, sel1 = self.manual_without_rfe()
            y_pred1 = model1.predict(xgb.DMatrix(X_test1))
            y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
            y_pred_orig1 = np.exp(y_pred1) if self.log_transformation_needed else y_pred1
            r21 = r2_score(y_test_orig, y_pred_orig1)
            mae1 = mean_absolute_error(y_test_orig, y_pred_orig1)
            rmse1 = root_mean_squared_error(y_test_orig, y_pred_orig1)
            print(f"R2: {r21:.4f}")
            self.results["Manual without RFE"] = (r21, mae1, rmse1, model1, X_test1, y_test, sel1, y_pred1)

            print("\n=== 2. Manual with RFE ===")
            model2, X_test2, y_test, sel2 = self.manual_with_rfe()
            y_pred2 = model2.predict(xgb.DMatrix(X_test2))
            y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
            y_pred_orig2 = np.exp(y_pred2) if self.log_transformation_needed else y_pred2
            r22 = r2_score(y_test_orig, y_pred_orig2)
            mae2 = mean_absolute_error(y_test_orig, y_pred_orig2)
            rmse2 = root_mean_squared_error(y_test_orig, y_pred_orig2)
            print(f"R2: {r22:.4f}")
            self.results["Manual with RFE"] = (r22, mae2, rmse2, model2, X_test2, y_test, sel2, y_pred2)

            print("\n=== 3. Automated (Optuna) without RFE ===")
            model3, X_test3, y_test, sel3 = self.automated_without_rfe()
            y_pred3 = model3.predict(xgb.DMatrix(X_test3))
            y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
            y_pred_orig3 = np.exp(y_pred3) if self.log_transformation_needed else y_pred3
            r23 = r2_score(y_test_orig, y_pred_orig3)
            mae3 = mean_absolute_error(y_test_orig, y_pred_orig3)
            rmse3 = root_mean_squared_error(y_test_orig, y_pred_orig3)
            print(f"R2: {r23:.4f}")
            self.results["Automated without RFE"] = (r23, mae3, rmse3, model3, X_test3, y_test, sel3, y_pred3)

            print("\n=== 4. Automated (Optuna) with RFE ===")
            model4, X_test4, y_test, sel4 = self.automated_with_rfe()
            y_pred4 = model4.predict(xgb.DMatrix(X_test4))
            y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
            y_pred_orig4 = np.exp(y_pred4) if self.log_transformation_needed else y_pred4
            r24 = r2_score(y_test_orig, y_pred_orig4)
            mae4 = mean_absolute_error(y_test_orig, y_pred_orig4)
            rmse4 = root_mean_squared_error(y_test_orig, y_pred_orig4)
            print(f"R2: {r24:.4f}")
            self.results["Automated with RFE"] = (r24, mae4, rmse4, model4, X_test4, y_test, sel4, y_pred4)

        def build_comparative(self):
            comparative_data = {
                'model': list(self.results.keys()),
                'r2': [self.results[k][0] for k in self.results],
                'mae': [self.results[k][1] for k in self.results],
                'rmse': [self.results[k][2] for k in self.results],
                'num_features': [len(self.results[k][6]) for k in self.results]
            }
            self.comparative_df = pd.DataFrame(comparative_data)
            self.comparative_df = self.comparative_df.sort_values(by='r2', ascending=False).reset_index(drop=True)

        def select_best(self):
            self.best_name = self.comparative_df.iloc[0]['model']
            self.best_r2, self.best_mae, self.best_rmse, self.model, self.X_test_selected, self.y_test, self.selected_features, self.y_pred_test = self.results[self.best_name]
            self.y_test_orig = np.exp(self.y_test) if self.log_transformation_needed else self.y_test
            self.y_pred_orig = np.exp(self.y_pred_test) if self.log_transformation_needed else self.y_pred_test

        def optimize(self):
            self.run_all()
            self.build_comparative()
            self.select_best()
            
            # Compute feature importance for the best model
            importance_gain = self.model.get_score(importance_type="gain")
            if importance_gain:
                total_gain = sum(importance_gain.values())
                normalized_gain = {feat: gain / total_gain for feat, gain in importance_gain.items()}
                self.best_features_df = pd.DataFrame({
                    "feature": list(normalized_gain.keys()),
                    "importance_gain_normalized": list(normalized_gain.values()),
                }).sort_values(by="importance_gain_normalized", ascending=False)
                self.best_features_df["importance_rank"] = range(1, len(self.best_features_df) + 1)
                self.best_features_df = self.best_features_df.set_index("importance_rank")
            else:
                self.best_features_df = pd.DataFrame()
            
            return {
                'comparative_df': self.comparative_df,
                'best_name': self.best_name,
                'best_r2': self.best_r2,
                'best_mae': self.best_mae,
                'best_rmse': self.best_rmse,
                'model': self.model,
                'X_test_selected': self.X_test_selected,
                'y_test': self.y_test,
                'selected_features': self.selected_features,
                'y_pred_test': self.y_pred_test,
                'y_test_orig': self.y_test_orig,
                'y_pred_orig': self.y_pred_orig,
                'base_mean': self.base_mean,
                'baseline_r2': self.baseline_r2,
                'baseline_mae': self.baseline_mae,
                'baseline_rmse': self.baseline_rmse,
                'best_features_df': self.best_features_df
            }

        def print_results(self):
            print("\n=== Comparative Model Results ===")
            print(self.comparative_df.to_string(index=False))

            print("\nBaseline (mean prediction):")
            print(f"R2: {self.baseline_r2:.4f}, MAE: {self.baseline_mae:.4f}, RMSE: {self.baseline_rmse:.4f}")

            print("\n" + "="*50)
            print(f"BEST MODEL: {self.best_name}")
            print(f"Best Test R2: {self.best_r2:.4f}")
            print("="*50)

            print("\nSelected Features:")
            print(self.selected_features)

            print(f"\nBase mean on train set: {self.base_mean:.4f}")

            print(f'R2 on test set (best model): {self.best_r2:.4f}\n')

    class MetricsComputer:
        def __init__(self, y_test_orig, y_pred_orig, base_mean):
            self.y_test_orig = y_test_orig
            self.y_pred_orig = y_pred_orig
            self.base_mean = base_mean

        def compute_metrics(self):
            percentiles = [99] + list(range(95, 0, -5)) + [1]
            table_rows = []
            for p in percentiles:
                cutoff = np.percentile(self.y_pred_orig, p)
                mask = self.y_pred_orig >= cutoff
                if not np.any(mask):
                    continue
                sub_pred = self.y_pred_orig[mask]
                sub_actual = self.y_test_orig[mask]
                avg_pred = np.mean(sub_pred)
                avg_actual = np.mean(sub_actual)
                mae_val = mean_absolute_error(sub_actual, sub_pred)
                rmse_val = root_mean_squared_error(sub_actual, sub_pred)
                lift = avg_actual / self.base_mean if self.base_mean > 0 else 0
                table_rows.append({
                    'percentile': f'P{p}',
                    'cutoff': cutoff,
                    'avg_pred': avg_pred,
                    'avg_actual': avg_actual,
                    'mae': mae_val,
                    'rmse': rmse_val,
                    'lift': lift,
                })
            metrics_df = pd.DataFrame(table_rows).set_index('percentile')
            return metrics_df.round(4)

    # Example usage
    maximizer = R2Maximizer(tabular_data_df, features, 'target')
    results = maximizer.optimize()
    maximizer.print_results()

    print("\n=== best_features_df (Top Features by Gain) ===")
    print(results['best_features_df'].to_string(float_format="{:.4f}".format))

    metrics_comp = MetricsComputer(results['y_test_orig'], results['y_pred_orig'], results['base_mean'])
    metrics_df = metrics_comp.compute_metrics()
    print("\n=== Performance by Percentile Threshold (Best Model) ===")
    print(metrics_df.to_string())

    print("\nNOTE:")
    print("- 'Pxx' means selecting samples with predicted value >= xx-th percentile (i.e., top (100-xx)%)")
    print("- Higher percentile = stricter threshold = higher lift, lower count")
    print("- Lift = avg_actual / base_mean")

    # Demonstrate making a prediction
    example_row = maximizer.tabular_data_df.iloc[0][results['selected_features']]
    dexample = xgb.DMatrix(pd.DataFrame([example_row]))
    example_pred = results['model'].predict(dexample)[0]
    example_pred = np.exp(example_pred) if maximizer.log_transformation_needed else example_pred

    print(f"\n=== Example Prediction ===")
    print(f"Input features:\n{results['selected_features']}")
    print(f"Predicted target: {example_pred:.4f}")


#### Lesson 9: Multi-Class Classification Intuition

    # 1. Metrics
    #!----------

    # 1.1. Baseline Accuracy (Majority Class Baseline)
    # The baseline accuracy is the accuracy you'd get by always predicting the most common 
    # class in the test data. It's the benchmark for 'doing nothing smart'-if you ignored 
    # all features and just used the majority class as your prediction for every instance, 
    # this would be your performance level. It's crucial for calculating relative 
    # improvements (e.g., how much better your model is than this naive approach). In 
    # multi-class classification, the class distribution (base rates) acts like the 
    # 'variance' equivalent, and metrics like accuracy or F1 compare your model's 
    # performance to this baseline.

    # 1.2. Accuracy
    # Accuracy measures the proportion of correct predictions out of all predictions.
    # Formula: 
    accuracy = number_of_correct_predictions / total_number_of_predictions
    # Intuition: Simple and intuitive-e.g., 80% accuracy means you get 8 out of 10 right. 
    # However, it's misleading in imbalanced datasets (e.g., if 90% of data is one class, 
    # predicting that class always gives 90% accuracy but ignores minorities). Use when 
    # classes are balanced and all errors cost the same, like in image recognition for 
    # everyday objects.

    # 1.3. Precision (Macro-Averaged)
    # Precision measures, for each class, the proportion of predicted positives that are 
    # actually positive, then averages across classes (macro treats each class equally).
    # Formula (per class): 
    precision_class = true_positives_class / (true_positives_class + false_positives_class)
    macro_precision = average(precision_class for each class)
    # Intuition: High precision means few false alarms-your 'yes' predictions are reliable. 
    # Low precision indicates many wrong positives. Use macro when all classes matter 
    # equally, even if imbalanced (e.g., rare disease detection where false positives 
    # waste resources).

    # 1.4. Recall (Macro-Averaged)
    # Recall measures, for each class, the proportion of actual positives that are correctly 
    # predicted, then averages across classes.
    # Formula (per class): 
    recall_class = true_positives_class / (true_positives_class + false_negatives_class)
    macro_recall = average(recall_class for each class)
    # Intuition: High recall means you catch most true cases-few misses. Low recall means 
    # many actual positives slip through. Critical when missing a class is costly (e.g., 
    # fraud detection across multiple types where missing any hurts).

    # 1.5. F1-Score (Macro-Averaged)
    # F1 is the harmonic mean of precision and recall, balancing both.
    # Formula (per class): 
    f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)
    macro_f1 = average(f1_class for each class)
    # Intuition: Useful when you need a single metric that penalizes extremes (low precision 
    # or low recall). Macro-F1 ensures minority classes aren't ignored. Aim for high F1 in 
    # imbalanced multi-class problems, like sentiment analysis with rare emotions.

    # 1.6. ROC AUC (One-vs-Rest, OVR)
    # ROC AUC measures the model's ability to distinguish between classes, averaged over 
    # one-vs-rest (treat each class as positive vs. all others).
    # Intuition:
    # - AUC = 1.0 -> Perfect separation: model ranks all positives above negatives.
    # - AUC = 0.5 -> No better than random guessing.
    # - AUC < 0.5 -> Worse than random (invert predictions?).
    # Business takeaway: AUC > 0.8 often means a strong model for many applications, but 
    # context matters (e.g., in noisy multi-class like customer segmentation, >0.7 might 
    # be decent). It's probability-based, so great for ranking or confidence-aware decisions.

    # 1.7. Log Loss (Multi-Class Cross-Entropy)
    # Log Loss penalizes wrong predictions based on predicted probabilities—harsh on confident 
    # wrongs.
    # Formula: 
    log_loss = - (1 / N) * sum(y_true * log(y_pred) for each sample and class)
    # Intuition: Low log loss means predictions are not just correct but confidently correct. 
    # High log loss indicates overconfidence in errors. Use when probabilities matter, like 
    # in betting or risk scoring across multiple outcomes.

    # 1.8. Confusion Matrix
    # A table showing actual vs. predicted classes, highlighting errors (off-diagonals).
    # Intuition: Visualizes where the model confuses classes (e.g., mistaking class 1 for 2). 
    # Diagonal = correct; rows = actuals, columns = predictions. Essential for debugging 
    # multi-class issues, like in medical diagnosis where confusing benign/malignant is worse 
    # than others.

    # 2. Real World Impact Test
    #!-------------------------

    # 1. Do we have a 'valid' model?
    # - if AUC >= 0.8: yes 
    # - else if AUC >= 0.6: yes (implies features set is noisy/imbalanced)
    # - else AUC < 0.6: hard no. Too weak—back to the drawing board.

    # 2. Does the model address the business problem?
    # In most real-world multi-class classification applications—like customer segmentation, 
    # defect type detection, sentiment labeling, or disease classification-the goal is 
    # reliable categorization under uncertainty, not perfect accuracy. We deploy models to 
    # guide decisions, balancing precision (avoid false positives), recall (catch true cases), 
    # and coverage (how many instances we confidently classify).
    # Key trade-offs:
    # - Precision (avoid false alarms): Critical when wrong positives are costly (e.g., 
    #   mislabeling a customer segment leads to wasted marketing).
    # - Recall (minimize misses): Vital when missing a class hurts (e.g., failing to detect 
    #   a rare defect type in manufacturing).
    # - Confidence thresholding: Use predicted probabilities to 'abstain' on low-confidence 
    #   cases—trade coverage for higher precision/recall on confident subsets. Useful in 
    #   high-stakes scenarios (e.g., only act on P90+ confidence predictions).
    # In practice, focus on per-class errors via confusion matrix and macro metrics to ensure 
    # fairness across classes.
    # The actionables can be summarized as follows:
    #!-----------------------------------------------------------------------------------
    #!        goal |    metric_focus |       does_it_address_business_problem_threshold |
    #!-----------------------------------------------------------------------------------
    #!   precision | macro_precision | macro_precision > 0.6, macro_F1 > 0.5, AUC > 0.8 |
    #!      recall |    macro_recall |    macro_recall > 0.6, macro_F1 > 0.5, AUC > 0.8 |
    #!  robustness |     CV macro_F1 | CV macro_F1 > 0.5, drop <10% from train macro_F1 |
    #!-----------------------------------------------------------------------------------

    # - Confidence thresholding intuition: By setting a minimum probability cutoff (e.g., 
    #   at P90 percentile of max predicted probs), you classify only high-confidence cases, 
    #   boosting precision/recall at the cost of coverage. E.g., at P99, you might classify 
    #   only 1% of data but with near-perfect metrics-ideal for automation where humans 
    #   handle the rest.

#### Lesson 10: Multi-Class Classification Implementation

    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        accuracy_score,
    )
    from sklearn.feature_selection import RFE

    #! features = ['feat_0', 'feat_1', ... 'feat_19']
    #! print(tabular_data_df.head(10))
    #!            feat_0    feat_1  ...   feat_17   feat_18   feat_19  class
    #! user_id                      ...
    #! 0        0.025729  0.165038  ...  0.040788  0.031007  0.018873      2
    #! 1        0.054640  0.008674  ...  0.005934  0.066555  0.033492      2
    #! 2        0.006200  0.032563  ...  0.010392  0.002205  0.018747      2
    #! 3        0.026806  0.017243  ...  0.004190  0.024178  0.006708      2
    #! 4        0.120043  0.058937  ...  0.033674  0.001554  0.006892      2
    #! 5        0.001613  0.051109  ...  0.010431  0.112692  0.039155      2
    #! 6        0.081659  0.112238  ...  0.060182  0.022404  0.176855      1
    #! 7        0.171431  0.015151  ...  0.074897  0.014173  0.068047      2
    #! 8        0.031450  0.068625  ...  0.033538  0.189332  0.010147      2
    #! 9        0.017432  0.005033  ...  0.095035  0.091150  0.063252      1

    # 1. Compute base_rate for each class
    base_rates = y.value_counts(normalize=True).sort_index()
    print('Class base rates:')
    for cls, rate in base_rates.items():
        print(f'Class {cls}: {rate:.2%}')
    print()

    class AUCMaximizer:
        def __init__(self, tabular_data_df, features, target):
            self.tabular_data_df = tabular_data_df
            self.features = features
            self.target = target
            self.n_classes = self.tabular_data_df[self.target].nunique()
            self.n_features_to_select = 10
            self.n_trials = 30
            self.test_size = 0.2
            self.val_size = 0.2
            self.random_state = 42
            self.default_params = {
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
            self.num_boost_round = 200
            self.early_stopping_rounds = 20
            self.optuna_boost_round = 1000
            self.optuna_early_stopping = 20
            self.optuna_search_spaces = {
                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                'eta': {'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
            self.results = {}
            self.comparative_df = None
            self.best_name = None
            self.best_auc = None
            self.model = None
            self.X_test_selected = None
            self.y_test = None
            self.selected_features = None
            self.y_pred_test = None
            self.base_rates = None
            self.best_features_df = None

        def manual_without_rfe(self):
            X_train, X_test, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            
            selected_features = self.features  # No RFE, use all features
            
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            
            params = self.default_params.copy()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            return model, X_test, y_test, selected_features

        def manual_with_rfe(self):
            X_train, X_test, y_train, y_test = train_test_split(
                self.tabular_data_df[self.features],
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            
            # RFE for feature selection
            base_model = xgb.XGBClassifier(
                **self.default_params,
                random_state=self.random_state,
                enable_categorical=True,
            )
            rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            print("\n=== Selected Features from RFE (Manual) ===")
            print(selected_features)
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
            
            params = self.default_params.copy()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            return model, X_test_selected, y_test, selected_features

        def automated_without_rfe(self):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                self.tabular_data_df.drop(self.target, axis=1),
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y_train_full
            )
            
            selected_features = X_train.columns.tolist()  # No RFE, use all features
            
            # Define the objective function for Optuna
            def objective(trial):
                max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
                eta = trial.suggest_float('eta', 
                                          self.optuna_search_spaces['eta']['low'], 
                                          self.optuna_search_spaces['eta']['high'], 
                                          log=self.optuna_search_spaces['eta']['log'])
                subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
                colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
                params = {
                    'objective': 'multi:softprob',
                    'num_class': self.n_classes,
                    'eval_metric': 'mlogloss',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                }
                dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.optuna_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.optuna_early_stopping,
                    verbose_eval=False,
                )
                y_pred_val = model.predict(dval)
                auc = roc_auc_score(y_val, y_pred_val, multi_class='ovr')
                return auc
            
            # Create and optimize the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_params['objective'] = 'multi:softprob'
            best_params['num_class'] = self.n_classes
            best_params['eval_metric'] = 'mlogloss'
            
            print('Best hyperparameters found by Optuna (without RFE):')
            print(best_params)
            
            # Train the final model with best params on full train set
            dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            
            model = xgb.train(
                best_params,
                dtrain_full,
                num_boost_round=self.optuna_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            return model, X_test, y_test, selected_features

        def automated_with_rfe(self):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                self.tabular_data_df.drop(self.target, axis=1),
                self.tabular_data_df[self.target],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.tabular_data_df[self.target]
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y_train_full
            )
            
            # RFE for feature selection
            base_model = xgb.XGBClassifier(
                **self.default_params,
                random_state=self.random_state,
                enable_categorical=True,
            )
            rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            print("\n=== Selected Features from RFE (Automated) ===")
            print(selected_features)
            
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            X_train_full_selected = X_train_full[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Define the objective function for Optuna on selected features
            def objective(trial):
                max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
                eta = trial.suggest_float('eta', 
                                          self.optuna_search_spaces['eta']['low'], 
                                          self.optuna_search_spaces['eta']['high'], 
                                          log=self.optuna_search_spaces['eta']['log'])
                subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
                colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
                params = {
                    'objective': 'multi:softprob',
                    'num_class': self.n_classes,
                    'eval_metric': 'mlogloss',
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                }
                dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val_selected, label=y_val, enable_categorical=True)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.optuna_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.optuna_early_stopping,
                    verbose_eval=False,
                )
                y_pred_val = model.predict(dval)
                auc = roc_auc_score(y_val, y_pred_val, multi_class='ovr')
                return auc
            
            # Create and optimize the study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_params['objective'] = 'multi:softprob'
            best_params['num_class'] = self.n_classes
            best_params['eval_metric'] = 'mlogloss'
            
            print('Best hyperparameters found by Optuna (with RFE):')
            print(best_params)
            
            # Train the final model with best params on full train set with selected features
            dtrain_full = xgb.DMatrix(X_train_full_selected, label=y_train_full, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
            
            model = xgb.train(
                best_params,
                dtrain_full,
                num_boost_round=self.optuna_boost_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            return model, X_test_selected, y_test, selected_features

        def run_all(self):
            print("\n=== 1. Manual without RFE ===")
            model1, X_test1, y_test, sel1 = self.manual_without_rfe()
            y_pred1 = model1.predict(xgb.DMatrix(X_test1))
            auc1 = roc_auc_score(y_test, y_pred1, multi_class='ovr')
            print(f"AUC (ovr): {auc1:.4f}")
            self.results["Manual without RFE"] = (auc1, model1, X_test1, y_test, sel1, y_pred1)

            print("\n=== 2. Manual with RFE ===")
            model2, X_test2, y_test, sel2 = self.manual_with_rfe()
            y_pred2 = model2.predict(xgb.DMatrix(X_test2))
            auc2 = roc_auc_score(y_test, y_pred2, multi_class='ovr')
            print(f"AUC (ovr): {auc2:.4f}")
            self.results["Manual with RFE"] = (auc2, model2, X_test2, y_test, sel2, y_pred2)

            print("\n=== 3. Automated (Optuna) without RFE ===")
            model3, X_test3, y_test, sel3 = self.automated_without_rfe()
            y_pred3 = model3.predict(xgb.DMatrix(X_test3))
            auc3 = roc_auc_score(y_test, y_pred3, multi_class='ovr')
            print(f"AUC (ovr): {auc3:.4f}")
            self.results["Automated without RFE"] = (auc3, model3, X_test3, y_test, sel3, y_pred3)

            print("\n=== 4. Automated (Optuna) with RFE ===")
            model4, X_test4, y_test, sel4 = self.automated_with_rfe()
            y_pred4 = model4.predict(xgb.DMatrix(X_test4))
            auc4 = roc_auc_score(y_test, y_pred4, multi_class='ovr')
            print(f"AUC (ovr): {auc4:.4f}")
            self.results["Automated with RFE"] = (auc4, model4, X_test4, y_test, sel4, y_pred4)

        def build_comparative(self):
            comparative_data = {
                'model': list(self.results.keys()),
                'auc': [self.results[k][0] for k in self.results],
                'num_features': [len(self.results[k][4]) for k in self.results]
            }
            self.comparative_df = pd.DataFrame(comparative_data)
            self.comparative_df = self.comparative_df.sort_values(by='auc', ascending=False).reset_index(drop=True)

        def select_best(self):
            self.best_name = self.comparative_df.iloc[0]['model']
            self.best_auc, self.model, self.X_test_selected, self.y_test, self.selected_features, self.y_pred_test = self.results[self.best_name]
            self.base_rates = self.y_test.value_counts(normalize=True).sort_index()

        def optimize(self):
            self.run_all()
            self.build_comparative()
            self.select_best()
            
            # Compute feature importance for the best model
            importance_gain = self.model.get_score(importance_type="gain")
            if importance_gain:
                total_gain = sum(importance_gain.values())
                normalized_gain = {feat: gain / total_gain for feat, gain in importance_gain.items()}
                self.best_features_df = pd.DataFrame({
                    "feature": list(normalized_gain.keys()),
                    "importance_gain_normalized": list(normalized_gain.values()),
                }).sort_values(by="importance_gain_normalized", ascending=False)
                self.best_features_df["importance_rank"] = range(1, len(self.best_features_df) + 1)
                self.best_features_df = self.best_features_df.set_index("importance_rank")
            else:
                self.best_features_df = pd.DataFrame()
            
            return {
                'comparative_df': self.comparative_df,
                'best_name': self.best_name,
                'best_auc': self.best_auc,
                'model': self.model,
                'X_test_selected': self.X_test_selected,
                'y_test': self.y_test,
                'selected_features': self.selected_features,
                'y_pred_test': self.y_pred_test,
                'base_rates': self.base_rates,
                'best_features_df': self.best_features_df
            }

        def print_results(self):
            print("\n=== Comparative Model Results ===")
            print(self.comparative_df.to_string(index=False))

            print("\n" + "="*50)
            print(f"BEST MODEL: {self.best_name}")
            print(f"Best Test AUC (ovr): {self.best_auc:.4f}")
            print("="*50)

            print("\nSelected Features:")
            print(self.selected_features)

            print("\nClass base rates on test set:")
            for cls, rate in self.base_rates.items():
                print(f"Class {cls}: {rate:.4f}")

            print(f'\nAUC (ovr) on test set (best model): {self.best_auc:.4f}\n')

    class MetricsComputer:
        def __init__(self, y_test, y_pred_test, base_rates=None):
            self.y_test = y_test
            self.y_pred_test = y_pred_test
            self.base_rates = base_rates if base_rates is not None else y_test.value_counts(normalize=True).sort_index()
            self.n_classes = len(self.base_rates)
            self.preds_argmax = np.argmax(self.y_pred_test, axis=1)

        def compute_metrics(self):
            # Confusion matrix for full set
            cm = confusion_matrix(self.y_test, self.preds_argmax, labels=range(self.n_classes))
            
            # Custom metrics for full set
            precisions = []
            recalls = []
            f1s = []
            for i in range(self.n_classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            
            precision_macro = np.mean(precisions)
            recall_macro = np.mean(recalls)
            f1_macro = np.mean(f1s)
            accuracy = accuracy_score(self.y_test, self.preds_argmax)
            
            lifts = [precisions[i] / self.base_rates[i] if self.base_rates[i] > 0 and precisions[i] > 0 else 0 for i in range(self.n_classes)]
            
            # Build overall table
            table_rows = [{
                "metric": "macro_precision",
                "value": round(precision_macro, 4),
            }, {
                "metric": "macro_recall",
                "value": round(recall_macro, 4),
            }, {
                "metric": "macro_f1",
                "value": round(f1_macro, 4),
            }, {
                "metric": "accuracy",
                "value": round(accuracy, 4),
            }]
            
            for i in range(self.n_classes):
                table_rows.append({
                    "metric": f"class_{i}_precision",
                    "value": round(precisions[i], 4),
                })
                table_rows.append({
                    "metric": f"class_{i}_recall",
                    "value": round(recalls[i], 4),
                })
                table_rows.append({
                    "metric": f"class_{i}_f1",
                    "value": round(f1s[i], 4),
                })
                table_rows.append({
                    "metric": f"class_{i}_lift",
                    "value": round(lifts[i], 2),
                })
            
            overall_metrics_df = pd.DataFrame(table_rows).set_index('metric')
            
            # Confusion matrix DF
            cm_df = pd.DataFrame(cm, index=[f"actual_{i}" for i in range(self.n_classes)], columns=[f"pred_{i}" for i in range(self.n_classes)])
            
            # Percentile-based metrics
            percentiles = [99] + list(range(95, 0, -5)) + [1]
            results = []
            max_probs = np.max(self.y_pred_test, axis=1)
            for p in percentiles:
                cutoff = np.percentile(max_probs, p)
                confident_mask = max_probs >= cutoff
                num_classified = np.sum(confident_mask)
                if num_classified > 0:
                    y_test_conf = self.y_test[confident_mask]
                    preds_conf = self.preds_argmax[confident_mask]
                    cm_conf = confusion_matrix(y_test_conf, preds_conf, labels=range(self.n_classes))
                    precisions_conf = []
                    recalls_conf = []
                    f1s_conf = []
                    for i in range(self.n_classes):
                        tp = cm_conf[i, i]
                        fp = np.sum(cm_conf[:, i]) - tp
                        fn = np.sum(cm_conf[i, :]) - tp
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                        precisions_conf.append(precision)
                        recalls_conf.append(recall)
                        f1s_conf.append(f1)
                    precision_macro = np.mean(precisions_conf)
                    recall_macro = np.mean(recalls_conf)
                    f1_macro = np.mean(f1s_conf)
                    accuracy = accuracy_score(y_test_conf, preds_conf)
                else:
                    precision_macro = 0.0
                    recall_macro = 0.0
                    f1_macro = 0.0
                    accuracy = 0.0
                results.append({
                    'percentile': f'P{p}',
                    'cutoff_prob': round(cutoff, 4),
                    'precision_macro': round(precision_macro, 4),
                    'recall_macro': round(recall_macro, 4),
                    'f1_macro': round(f1_macro, 4),
                    'accuracy': round(accuracy, 4),
                })

            confidence_metrics_df = pd.DataFrame(results).set_index('percentile')
            
            return overall_metrics_df, cm_df, confidence_metrics_df

    # Example usage
    maximizer = AUCMaximizer(tabular_data_df, features, 'class')
    results = maximizer.optimize()
    maximizer.print_results()

    print("\n=== best_features_df (Top Features by Gain) ===")
    print(results['best_features_df'].to_string(float_format="{:.4f}".format))

    metrics_comp = MetricsComputer(results['y_test'], results['y_pred_test'], results['base_rates'])
    overall_metrics_df, cm_df, metrics_df = metrics_comp.compute_metrics()
    print("\n=== Overall Performance Metrics (Best Model) ===")
    print(overall_metrics_df.to_string())
    print("\n=== Confusion Matrix (Best Model) ===")
    print(cm_df.to_string())
    print("\n=== Performance by Percentile Threshold (Best Model) ===")
    print(metrics_df.to_string())
    #!              cutoff  avg_pred  avg_actual     mae    rmse    lift
    #! percentile
    #! P99         11.7430   13.5616     14.4697  2.1531  2.5311  4.2645
    #! P95          7.4885   10.0334     10.3819  1.7722  2.3128  3.0598
    #! P90          6.0153    8.3771      8.5573  1.4777  2.0037  2.5220
    #! P85          5.0194    7.3983      7.5583  1.2936  1.7732  2.2276
    #! P80          4.5000    6.7331      6.8971  1.2012  1.6453  2.0327
    #! P75          4.0965    6.2463      6.3980  1.0782  1.5103  1.8856
    #! P70          3.7181    5.8545      5.9804  1.0038  1.4160  1.7625
    #! P65          3.4268    5.5296      5.6495  0.9453  1.3410  1.6650
    #! P60          3.1664    5.2518      5.3699  0.8964  1.2798  1.5826
    #! P55          2.9324    5.0068      5.1196  0.8466  1.2219  1.5089
    #! P50          2.6792    4.7851      4.8913  0.8065  1.1723  1.4416
    #! P45          2.4735    4.5840      4.6856  0.7720  1.1305  1.3810
    #! P40          2.2869    4.3997      4.5048  0.7424  1.0924  1.3277
    #! P35          2.0921    4.2303      4.3323  0.7168  1.0598  1.2768
    #! P30          1.8960    4.0704      4.1702  0.6900  1.0272  1.2290
    #! P25          1.7457    3.9202      4.0211  0.6659  0.9982  1.1851
    #! P20          1.5851    3.7794      3.8745  0.6408  0.9701  1.1419
    #! P15          1.4365    3.6457      3.7361  0.6178  0.9441  1.1011
    #! P10          1.2299    3.5174      3.6067  0.5964  0.9208  1.0630
    #! P5           1.0253    3.3921      3.4766  0.5748  0.8977  1.0246
    #! P1           0.6922    3.2899      3.3708  0.5572  0.8800  0.9935

    print("\nNOTE:")
    print("- Metrics include macro averages and per-class precision, recall, f1, lift")
    print("- Lift = precision / base_rate for each class")
    print("- 'Pxx' means selecting samples with max predicted probability >= xx-th percentile")
    print("- Higher percentile = stricter threshold = higher precision_macro, lower coverage")

    # Demonstrate making a prediction
    example_row = tabular_data_df.iloc[0][results['selected_features']]
    dexample = xgb.DMatrix(pd.DataFrame([example_row]))
    example_pred = results['model'].predict(dexample)[0]

    print(f"\n=== Example Prediction ===")
    print(f"Input features:\n{results['selected_features']}")
    print(f"Predicted class probabilities: {example_pred}")
    print(f"Predicted class: {np.argmax(example_pred)}")


