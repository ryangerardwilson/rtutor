# The Conventional Core of Python

## Part I: Python Basics

### Lesson 1: Running the REPL

    $ python3
    >>> print("Hello World")
    Hello World
    >>> 1000 + 0.1 + 1 
    1001.1
    >>> _ + 1  # _ is last result, interactive only
    1002.1
    >>> quit()  # Or Ctrl+D to exit

### Lesson 2: Python Programs

    # hello.py
    #!/usr/bin/env python3
    print("Hello World")

    # Make executable 
    $ chmod +x hello.py
    $ ./hello.py
    Hello World

    # Or just:
    $ python3 hello.py

### Lesson 3: Primitives, Variables, and Expressions

    # Primitive Types & Variable Assignment
    x: int = 42  
    y: float = 3.14159
    z: str = "Hello World"
    a: bool = True

    # Expression
    result = 2 + 3 * 4  # 14

    # interest.py example
    principal = 1000
    rate = 0.05
    numyears = 5
    year = 1
    while year <= numyears:
        principal = principal * (1 + rate)
        print(f"{year:>3d}  {principal:0.2f}")
        year += 1

### Lesson 4A: Arithmetic Operators

    x = 10
    y = 3

    x + y    # 13
    x - y    # 7
    x * y    # 30
    x / y    # 3.333...
    x // y   # 3 (floor)
    x ** y   # 1000
    x % y    # 1
    -x       # -10
    +x       # 10

    abs(-42)              # 42
    divmod(10, 3)         # (3, 1)
    pow(2, 3, 5)          # 8 % 5 = 3
    round(3.14159, 2)     # 3.14

    a = 0b11001001  # 201
    mask = 0b11110000
    (a & mask) >> 4  # 12 (0b1100)

### Lesson 4B: Arithmetic Operators

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

### Lesson 5: Conditionals and Control Flow

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

### Lesson 6A: Text Strings

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

### Lesson 6B: Text Strings

    # Concat
    a + 'ly'  # 'Hello Worldly'

    # Conversion
    x = '37'
    y = '42'
    x + y               # '3742'
    int(x) + int(y)     # 79

    num = 12.34567
    str(num)            # '12.34567'
    repr('hello\nworld')# "'hello\\nworld'"
    format(num, '0.2f') # '12.35'
    f'{num:0.2f}'       # '12.35'

### Lesson 7: File Input and Output

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

    # Encoding
    with open('data.txt', encoding='latin-1') as file:
        data = file.read()

### Lesson 8: Lists

    names = ['Dave', 'Paula', 'Thomas', 'Lewis']
    names[2]          # 'Thomas'
    names[2] = 'Tom'  # Modify
    names[-1]         # 'Lewis'
    names.append('Alex')
    names.insert(2, 'Aya')
    for name in names: print(name)
    b = names[0:2]  # ['Dave', 'Paula']
    names[0:2] = ['Dave', 'Mark', 'Jeff']  # Replace slice
    combo = ['x', 'y'] + ['z']  # ['x', 'y', 'z']

    empty = []
    letters = list('Dave')  # ['D', 'a', 'v', 'e']
    mixed = [1, 'Dave', 3.14, ['Mark', 7, 9, [100, 101]], 10]
    mixed[3][2]     # 9
    mixed[3][3][1]  # 101

    # pcost.py 
    import sys
    if len(sys.argv) != 2:
        raise SystemExit(f'Usage: {sys.argv[0]} filename')

    rows = []
    with open(sys.argv[1], 'rt') as file:
        for line in file:
            rows.append(line.strip().split(','))

    total = sum([int(row[1]) * float(row[2]) for row in rows])
    print(f'Total cost: {total:0.2f}')

### Lesson 9: Tuples

    holding = ('GOOG', 100, 490.10)
    address = ('www.python.org', 80)

    empty = ()          # 0-tuple
    single = ('item',)  # 1-tuple

    name, shares, price = holding  # Unpack
    portfolio = []
    with open('portfolio.csv') as file:
        for line in file:
            row = line.strip().split(',')
            name = row[0]
            shares = int(row[1])
            price = float(row[2])
            holding = (name, shares, price)
            portfolio.append(holding)

    portfolio[0]      # ('AA', 100, 32.2)
    portfolio[1][1]   # 50

    total = 0.0
    for name, shares, price in portfolio:
        total += shares * price

    # Comprehension
    total = sum([shares * price for _, shares, price in portfolio])
