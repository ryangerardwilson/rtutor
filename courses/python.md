# The Conventional Core of Python

## Lesson 1: Hello World

    #!/usr/bin/env python3
    print("Hello, world!")

## Lesson 2: Built-In Types

    # Python doesn't have "primitives" like C does—everything's an object here
    # But these are the core built-in types that act as the basics. 

    # Integer (int): Whole numbers, unlimited size in Python 3. 
    x: int = 42
    # Float (float): Floating-point numbers, equivalent to C doubles 
    y: float = 3.14159
    # String (str): Text, immutable sequences of Unicode characters 
    z: str = "Linus was here"
    # Boolean (bool): True or False, subclass of int 
    a: bool = True
    # NoneType: The singleton None, for representing absence of value
    b: None = None
    # Complex: For complex numbers, real + imaginary parts 
    c: complex = 1 + 2j
    # Bytes: Immutable sequences of bytes, for binary data 
    d: bytes = b'raw bytes'
    # Bytearray: Mutable version of bytes 
    e: bytearray = bytearray(b'mutable bytes')

## Lesson 3: While Loop & Pretty Formatting

    principal = 1000
    rate = 0.05
    numyears = 5
    year = 1

    while year <= numyears:
        principal = principal * (1+rate)
        print(f"{year:>3d}  {principal:0.2f}")
        year += 1

## Lesson 4: Arithmetic Operators

    x+y
    x-y
    x*y
    x/y
    x//y # Truncating Division
    x**y # Power
    x%y # Modulo (x mod y)
    -x # Unary minus
    +x # Unary plus

## Lesson 5: Common Mathematical Functions

    abs(x) # Absolute value
    divmod(x,y) # Returns (x // y, x % y)
    pow(x,y [,modulo]) # Returns (x ** y) % modulo
    round(x,[n]) # Rounds to the nearest multiple of 10 to the nth power

## Lesson 6: Bit Manipulation Operators

    x << y # Left shift
    x >> y # Right shift
    x & y # Bitwise and
    x | y # Bitwise or
    x ^ y # Bitwise xor (exclusive or)
    ~x # Bitwise negation

    # Usage with binary integers
    a = 0b11001001
    mask = 0b11110000
    x = (a & mask) >> 4 # x = 0b1100 (12)

## Lesson 7: Comparison Operators

    x == y
    x != y
    x < y
    x > y
    x >= y
    x <= y

## Lesson 8: Logical Operators

    x or y
    x and y
    not x

## Lesson 9: Shorthand Operator Syntax

    x = x + 1
    x += 1

    y = y * n
    y *= n

## Lesson 10: Conditionals

    if s == '.htm':
        content = 'text/html'
    elif s == '.png':
        pass
    else:
        raise RuntimeError(f"Unknown content type {s!r}")

## Lesson 11: Conditioned Expression

    maxval = a if a > 5 else b

## Lesson 12: Walrus Operator

    # Assigns and evaluates condition simultaneously
    x = 0
    while (x := x + 1) < 10:
        print(x)
