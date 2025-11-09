# C

## Part I: K&R 2nd Ed Concepts

### Section A: K&R Chapter 1

#### Lesson 1: Hello World

    #include <stdio.h>

    int main() {
        printf("Hello, world!");
        return 0;
    }

#### Lesson 2: Compile & Execute

    clang main.c -o main && ./main

#### Lesson 3: Basic Types

    #include <stdio.h>

    int main() {
        int a = 5;
        char s[10] = "Linus";
        return 0;
    }

#### Lesson 4: Basic Data Types and Their Qualifiers

    #include <stdio.h>

    int main() {

        // The 4 Complete Basic Data Types 
        // Complete means -> Compiler allocates memory 
        // based on type declaration.
        int i = 1337;              
        float f = 3.14f;           
        double d = 3.1415926535;   
        char c = 'L';             
        // Equivalent:
        char c2 = 76;

        return 0;
    } 

#### Lesson 5: Derived Data Types

    #include <stdio.h>

    int main() {

        // Array: A contiguous block of the same complete type
        // Stings are arrays derived from chars, the stdlib assumes this set up
        char name[10] = "Linus";  // Ascii bytes: L,i,n,u,s,\0,\0,\0,\0,\0
        // Derived from int
        int numbers[5] = {1, 2, 3, 4, 5};
        // Derived from float
        float decimals[4] = {3.14, 2.718, 1.618, 0.577};

        return 0;
    } 

#### Lesson 6: Formatting with Printf

    #include <stdio.h>

    int main() {

        char name[10] = "Linus";
        int age = 55;
        float height = 5.11f;
        double pi = 3.14159;
        long net_worth = 1000000000000L;

        // %s for strings: null-terminated or segfault
        printf("Name: %s\n", name);
        // %d for decimal integers.
        printf("Age: %d\n", age);
        // Minimum width 3, pads with spaces. Useful for alignment.
        printf("Width: %3d\n", age);
        // %f for floats, default 6 decimal places--ugly as sin.
        printf("Height: %f\n", height);
        // Precision: 2 digits after decimal.
        printf("Prec: %.2f\n", height);

        return 0;
    } 

#### Lesson 7: Basic Arithmetic Expressions

    #include <stdio.h>

    int main() {
        double result;
        result = 2 * (2 + 3) / 4.0 - 1;
        printf("%f\n", result);
        return 0;
    } 

#### Lesson 8: If-Else

    #include <stdio.h>

    int main() {
        int age = 25;
        if (age >= 18) {
            printf("Adult\n");
        } else {
            printf("Kid\n");
        }
        return 0;
    } 

#### Lesson 9: While Loop

    #include <stdio.h>

    int main() {
        char s[] = "hello";
        int i = 0;
        while (s[i] != '\0') {
            i++;
        }
        printf("Length: %d\n", i);
        return 0;
    } 

#### Lesson 10: For Loop - Repeat Without Being a Loop Idiot

    #include <stdio.h>

    int main() {
        int i;
        for (i = 0; i < 10; i++) {
            printf("%d\n", i * i);
        }
        return 0;
    } 

#### Lesson 11: Getchar, Putchar, EOF

    #include <stdio.h>

    int main() {
        int c;
        printf("Type text, Ctrl+D to flush buffer"); 
        printf("OR: Enter, Ctrl+D to trigger EOT, which c interprets as EOF"); 

        // NOTE: EOF is
        // 1. not a character, but a macro defined in stdio.h as -1. 
        // 2. what getchar() spits back when there's no more input to read 
        while ((c = getchar()) != EOF) {
            putchar(c);
        }
        return 0;
    }

#### Lesson 12: scanf

    #include <stdio.h>

    int main() {
        char name[20];
        printf("Enter name: ");
        scanf("%s", name);
        printf("Hello, %s\n", name);
        return 0;
    } 

#### Lesson 13: Constants, Mutables, Globals & Functions

    #include <stdio.h>

    #define INC 1 // Preprocessor constant.
    const int G_CONST = 42; // Global const.

    int global = 0; // Mutable global: mostly bad.

    int add(int a, int b) {
        const int L_CONST = 10; // Local const.
        int sum = a + b + L_CONST;
        global += INC;
        return sum;
    }

    int main() {
        printf("%d\n", add(3, 4)); 
        {
            const int b_const = G_CONST; // Block const.
            printf("Block: %d\n", b_const);
        }
        printf("Global: %d\n", global); // 1
        return 0;
    }

### Section B: K&R Chapter 2

#### Lesson 1: Variable Names

    #include <stdio.h>

    int main() {
        int valid_best_practice_name = 42; // Starts with letter; lower case;
uses underscores
        int _underscore_start = 43; // OK, but avoid double `__`, as it may 
                                    // conflict with lib methods
        int name_with_digits123 = 44; // Digits fine after first char
        int lower_case = 1; // Case matters
        int LOWER_CASE = 2; // Different from above
        return 0;
    }

#### Lesson 2A: Data Types and Sizes

    // There are only a few basic data types in C: 
    // char:    a single byte, capable of holding one character in the local 
    //          character set
    // int:     an integer, typically reflecting the natural size of integers 
    //          on the host machine
    // float:   single-precision floating point
    // double:  double-precision floating point

    // Integer Qualifiers
    short int sh; // atleast 16 bits
    long int counter; // atleast 32 bits

    // Integer/ Char Qualifiers
    signed int si; // default, even if signed qualifier is not specified
    unsigned int usi; // always positive, and holds 2x more positive integers
    signed char sc; // maps the 256 ASCII values to the int range -128 to 127
    unsigned char usc; // has ASCII decimal values from 0 to 255

    // Float/ Double Qualifiers
    long double ld; // extended-precision floating point    

#### Lesson 2B: Data Types and Sizes

    // limits.h and float.h contain symbolic constants for the sizes allocated
    // by types and their qualifiers
    #include <stdio.h>
    #include <limits.h>  // For INT_MAX etc.
    #include <float.h>   // For FLT_MAX etc.

    int main() {
        // %zu is the C99 specifier for printing size_t (unsigned, from 
        // sizeof); use it to avoid UB with %d on large values
        printf("char: %zu bytes\n", sizeof(char));  // 1, always.
        printf("int: %zu bytes\n", sizeof(int));    // 4, probably.
        printf("long: %zu bytes\n", sizeof(long));  // 8 on 64-bit.
        printf("Unsigned int max: %u\n", UINT_MAX);
        printf("Float max: %e\n", FLT_MAX);     // Max float value, 
                                                // in scientific notation
        return 0;
    }

#### Lesson 3A: Constants & Qualifiers

    #include <stdio.h> 

    int main() {
        // Integer constants (literals): decimal by default, octal (0 prefix), 
        // hex (0x prefix).
        int dec = 42;   
        int oct = 052;  
        int hex = 0x2A; 

        return 0;
    }


#### Lesson 3B: Integer Constants with Suffixes

    #include <stdio.h>

    int main() {
        // Look, in C, integers are ints by default, but you can force them to be long or unsigned.
        // Don't be sloppy; use suffixes to avoid surprises on different machines.
        long big_num = 123456789L;  // L for long, because who wants overflow headaches?
        unsigned int positive_only = 42U;  // U for unsigned, doubles your positive range.
        unsigned long even_bigger = 0xFUL;  // Hex with UL suffix, value 15 decimal. Hex is for humans who hate octal.

        printf("Long: %ld, Unsigned: %u, Unsigned Long: %lu\n", big_num, positive_only, even_bigger);
        // Octal? Prefix with 0. Like 037 is 31 decimal. But seriously, use hex instead unless you're punishing yourself.
        int octal = 037;
        printf("Octal 037 as decimal: %d\n", octal);

        return 0;
    }

#### Lesson 4: Floating-Point Constants

    #include <stdio.h>

    int main() {
        // Floats and doubles: decimals or exponents. Double by default, because precision matters, dammit.
        double pi_approx = 3.14159;  // Decimal point makes it floating-point.
        float small_pi = 3.14159F;   // F suffix for float; use when memory is tight, but don't cry about lost precision.
        long double super_precise = 1.23456789e-10L;  // e for exponent, L for long double. For when double isn't enough.

        printf("Double: %.5f, Float: %.5f, Long Double: %.10Lf\n", pi_approx, small_pi, super_precise);
        // Pro tip: Avoid floats unless you're optimizing arrays. Doubles are the way to go on modern hardware.

        return 0;
    }

#### Lesson 5: Character Constants and Escape Sequences

    #include <stdio.h>

    int main() {
        // Characters are just ints in disguise. 'x' is the ASCII value of x, not the string "x".
        char letter = 'x';  // Single quotes. Value? Probably 120 in ASCII.
        char null_char = '\0';  // Null terminator. Numeric value 0, but write it as char for clarity.
        char bell = '\a';  // Escape for alert (beep). Try it; annoy your terminal.
        char newline = '\n';  // Newline, obviously.
        char tab = '\t';  // Horizontal tab.

        // Octal escapes: '\ooo' for arbitrary bytes.
        char vtab_oct = '\013';  // Vertical tab in octal.
        // Hex escapes: '\xhh' for the same, but hex is saner.
        char bell_hex = '\x7';  // Bell in hex.

        printf("Char value: %d\n", letter);  // Prints the int value.
        printf("Escape demo: Alert\a Newline\n Tab\t Done.\n");
        // Backslash itself? '\\'. Quotes? '\'' or '\"'. Don't mess this up or your strings break.

        // Full escapes: \a \b \f \n \r \t \v \\ \? \' \" \ooo \xhh
        // Use them in strings too, like "Hello\nWorld".

        return 0;
    }

#### Lesson 6: String Constants and Concatenation

    #include <stdio.h>
    #include <string.h>  // For strlen, because rolling your own is fine but lazy is better.

    int main() {
        // Strings: Double quotes, null-terminated arrays under the hood.
        char greeting[] = "Hello, world";  // Includes '\0' at end. Storage: length + 1.
        char empty[] = "";  // Just '\0'.

        // Concatenation at compile time: Split long strings.
        char long_str[] = "This is a long string that "
                          "spans multiple lines. Neat, huh?";

        printf("%s\n", long_str);
        // Length? Use strlen, excludes '\0'.
        printf("Length of greeting: %zu\n", strlen(greeting));  // 12, not 13.

        // Don't confuse 'x' (int) with "x" (array with 'x' and '\0').
        // 'x' + 1 is 'y' maybe, but "x"[0] is 'x'.

        // Custom strlen if you're masochistic:
        int my_strlen(const char s[]) {
            int i = 0;
            while (s[i] != '\0') ++i;
            return i;
        }
        printf("My strlen: %d\n", my_strlen(greeting));

        return 0;
    }

#### Lesson 7: Enumeration Constants

    #include <stdio.h>

    int main() {
        // Enums: Named int constants. Better than #define for related values.
        enum boolean { NO, YES };  // NO=0, YES=1 by default.
        enum months { JAN = 1, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC };  // Starts at 1, increments.

        // Partial values: Gaps? Fine.
        enum escapes { BELL = '\a', BACKSPACE = '\b', TAB = '\t', NEWLINE = '\n', VTAB = '\v', RETURN = '\r' };

        printf("YES: %d, FEB: %d, BELL: %d\n", YES, FEB, BELL);
        // Use in vars: enum boolean flag = YES;
        // Compilers might not enforce values, but debuggers love symbolic names.
        // Alternative to #define, auto-generates values. Portable and readable.

        // Distinct enums can't share names, but values can overlap.

        return 0;
    }

#### Lesson 8: Constant Expressions

    #include <stdio.h>

    #define MAXLINE 1000  // Constant expression.

    int main() {
        // Constants only, evaluated at compile time.
        char line[MAXLINE + 1];  // Array size must be constant.

        // In arrays or wherever constants are needed.
        #define LEAP 1
        int days[31 + 28 + LEAP + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31];  // For leap year.

        printf("Size of line: %zu\n", sizeof(line));  // 1001 bytes.

        // Useful for portability; no runtime cost.

        return 0;
    }

#### Lesson 9: Declarations and Initializations

    #include <stdio.h>

    int main() {
        // Declare before use. Type + vars.
        int lower, upper = 100, step;  // Multiple, some initialized.
        char c = 'a';
        char line[1000] = {0};  // Zero-init array.

        // Init: = expression.
        int i = 0;
        int limit = MAXLINE + 1;  // From #define.
        float eps = 1.0e-5;

        // Externals/statics: Init once, to constant. Autos: Each entry, any expr. Uninit autos: Garbage.
        // Externals/statics default to 0.

        printf("Upper: %d, Eps: %e\n", upper, eps);

        // Can split declarations:
        int foo;
        int bar;

        // Add comments per var if you want.

        return 0;
    }

#### Lesson 10: The Const Qualifier

    #include <stdio.h>
    #include <string.h>

    int main() {
        // Const: Can't change. For safety.
        const double e = 2.71828182845905;  // Try e = 3; compiler yells.
        const char msg[] = "warning: ";  // Array elements immutable.

        // In functions: Promises no change.
        // int strlen(const char[]);  // From string.h.

        printf("%s %f\n", msg, e);

        // Attempt to modify? Undefined behavior. Don't.

        return 0;
    }

#### Lesson 11: Arithmetic Operators

    #include <stdio.h>

    int main() {
        // +, -, *, /, % (modulus).
        int x = 10, y = 3;
        printf("Add: %d, Sub: %d, Mul: %d\n", x + y, x - y, x * y);
        printf("Div: %d (truncates), Mod: %d\n", x / y, x % y);  // 3 and 1.

        // % only for ints. No floats.
        // Negative? Machine-dependent. Avoid if portable.

        // Precedence: * / % > + -. Left to right.
        int res = 1 + 2 * 3;  // 7, not 9.

        // Unary + - higher.

        // Leap year example:
        int year = 2000;
        if ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)
            printf("%d is leap.\n", year);  // Yes.

        return 0;
    }

#### Lesson 12: Relational and Logical Operators

    #include <stdio.h>

    int main() {
        // Relational: > >= < <=. Same precedence.
        // Equality: == !=. Lower than relational.
        // All lower than arithmetic.
        int i = 5, lim = 10;
        if (i < lim - 1) printf("Yes.\n");  // i < (lim-1).

        // Logical: && ||. Short-circuit: Stops early.
        // && > || precedence. Lower than relational.
        int c = 'a';
        if (i < lim - 1 && (c = getchar()) != '\n' && c != EOF) {}  // From getline.

        // No parens needed for && chain.
        // But for assignment: (c = getchar()) != '\n'.

        // True=1, False=0.
        int valid = 1;
        if (!valid) printf("Not valid.\n");  // ! negates: Non-zero to 0.

        // Exercise 2-2: For loop without &&/||.
        // Equivalent to: for (i=0; i < lim-1 && (c=getchar()) != '\n' && c != EOF; ++i) s[i]=c;
        int ii = 0;
        while (ii < lim - 1) {
            c = getchar();
            if (c == '\n') break;
            if (c == EOF) break;
            // s[ii] = c;  // Assuming s array.
            ++ii;
        }

        return 0;
    }

## Part II: K&R 2nd Ed - Exercises

### Section A: Chapter 1 Exercises 1-12

#### Lesson 1: Hello World

    // Ex: Run the `hello, world` program on your system. 
    #include <stdio.h>

    int main(void) {
        printf("hello, world\n");
        return 0;
    }

#### Lesson 2: \c

    // Experiment to find out what happens when printf's argument string 
    // contains \c
    #include <stdio.h>

    int main(void) {
        printf("hello, world\c");
        return 0;
    }

#### Lesson 3: Temperature Conversion

    // Modify the temperature conversion program to print a heading above the 
    // table.
    #include <stdio.h>

    int main(void) {
        float fahr, celsius;
        int lower, upper, step;

        lower = 0;
        upper = 300;
        step = 20;

        // Printing a heading above the table
        printf("Fahr\tCelsius\n");
        printf("---------------\n");

        fahr = lower;
        while (fahr <= upper) {
            celsius = (5.0 / 9.0) * (fahr - 32.0);
            printf("%3.0f\t\t%6.1f\n", fahr, celsius);
            fahr = fahr + step;
        }
        return 0;
    }

#### Lesson 4: Celsius to Fahrenheit

    // Write a program to print the corresponding Celsius to Fahrenheit table.
    #include <stdio.h>

    int main(void) {
        float celsius, fahr;
        int lower, upper, step;

        lower = 0;
        upper = 300;
        step = 20;

        printf("Celsius\tFahr\n");
        printf("---------------\n");

        celsius = lower;
        while (celsius <= upper) {
            fahr = (9.0 / 5.0) * celsius + 32.0f;
            printf("%3.0f\t\t%6.1f\n", celsius, fahr);
            celsius = celsius + step;
        }

        return 0;
    }

#### Lesson 5: Temperature Conversion Reversed

    // Modify the temperature conversion program to print the table in reverse 
    // order, that is, from 300 degrees to 0.  
    #include <stdio.h>

    int main(void) {
        float celsius, fahr;
        int lower, upper, step;

        lower = 0;
        upper = 300;
        step = 20;

        printf("Celsius\tFahr\n");
        printf("---------------\n");

        for (celsius = upper; celsius >= lower; celsius = celsius - step) {
            fahr = (9.0 / 5.0) * celsius + 32.0f;
            printf("%3.0f\t\t%6.1f\n", celsius, fahr);
        }

        return 0;
    }

#### Lesson 6: getchar()

    // Verify that the expression getchar() != EOF is 0 or 1.
    #include <stdio.h>

    int main(void) {
        printf("value of expression: %d", getchar() != EOF);
        return 0;
    }

#### Lesson 7: EOF

    // Write a program to print the value of EOF.
    #include <stdio.h>

    int main(void) {
        printf("EOF: %d", EOF);
        return 0;
    }

    // NOTE: The value of the EOF character is -1, which is an integer.

#### Lesson 8: Counting chars

    // Write a program to count blanks, tabs, and newlines.
    #include <stdio.h>

    int main(void) {
        int blanks_nr = 0;
        int tabs_nr = 0;
        int newlines_nr = 0;
        char c;
        while ((c = getchar()) != EOF) {
            if (c == ' '){ ++blanks_nr; } 
            else if (c == '\t'){ ++tabs_nr; } 
            else if (c == '\n'){ ++newlines_nr; }
        }
        printf(
            "blanks_nr: %d\ntabs_nr: %d\nnewlines_nr: %d\n", 
            blanks_nr, tabs_nr, newlines_nr);
        return 0;
    }

#### Lesson 9: Copy Input to Output I

    // Write a program to copy its input to its output, replacing each string 
    // of one or more blanks by a single blank.
    #include <stdio.h>

    int main(void) {
        int c;
        int last_c = '\0';
        while ((c = getchar()) != EOF) {
            if (c != ' ' || last_c != ' ') { putchar(c); }
            last_c = c;
        }
        return 0;
    }

#### Lesson 10: Copy Input to Output II

    // Write a program to copy its input to its output, replacing each tab by 
    // \t, each backspace by \b, and each backslash by \\. This makes tabs and 
    // backspaces visible in an unambiguous way.
    #include <stdio.h>

    int main(void) {
        char c;
        while ((c = getchar()) != EOF) {
            if (c == '\t'){ putchar('\\'); putchar('t'); } 
            else if (c == '\b') { putchar('\\'); putchar('b'); } 
            else if (c == '\\'){ putchar('\\'); putchar('\\'); } 
            else{ putchar(c); }
        }
        return 0;
    }

#### Lesson 11: Testing the Word Count Program

    // How would you test the word count program? What kinds of input are most 
    // likely to uncover bugs if there are any?
    #include <stdio.h>
    #define IN 1
    #define OUT 0

    int main(void) {
        char nl;
        char nw;
        char nc;
        int state;
        nl = nw = nc = 0;
        state = OUT;
        char c;
        while ((c = getchar()) != EOF) {
            ++nc;
            if (c == '\n'){ ++nl; }
            if (c == ' ' || c == '\n' || c == '\t'){ state = OUT; } 
            else if (state == OUT){ state = IN; ++nw; }
        }
        printf("lines: %d\nwords: %d\ncharacters: %d\n", nl, nw, nc);
        return 0;
    }

#### Lesson 12: Print Input

    // Write a program that prints its input one word per line.
    #include <stdio.h>

    int main() {
        int c; // current character
        int pc = EOF; // previous character

        while ((c = getchar()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\n') {
                if (pc != ' ' && pc != '\t' && pc != '\n') { putchar('\n'); }
            } else { putchar(c); }
            pc = c;
        }
        return 0;
    }

### Section B: Chapter 1 Exercises 13-19

#### Lesson 13A: Histogram of Lengths of Input Words

    // Write a program to print a histogram of the lengths of words in its 
    // input. It is easy to draw the histogram with the bars horizontal; 
    // a vertical orientation is more challenging.
    #include <stdio.h>
    #define TRUE 1
    #define FALSE 0
    #define BUFFER 100

    int main(void){
        int histogram[BUFFER];
        int histogram_length = 0;
        int max_word_count = 0;
        // Initialize the histogram array with 0
        int i;
        for (i = 0; i < BUFFER; ++i) { histogram[i] = 0; }
        char c;
        int word_count_index = 0;
        while ((c = getchar())){
            if (c == ' ' || c == '\t' || c == '\n' || c == EOF){
                if (word_count_index > 0){
                    ++histogram[word_count_index - 1];
                    if (histogram[word_count_index - 1] > max_word_count) { 
                        max_word_count = histogram[word_count_index - 1]; 
                    }
                    if (histogram_length < word_count_index - 1) { 
                        histogram_length = word_count_index - 1; 
                    }
                    word_count_index = 0;
                }
                if (c == EOF){ break; }
            } else{ ++word_count_index; }
        }
        // TBC

#### Lesson 13B: Histogram of Lengths of Input Words (Horizontal)

        // Add in the histogram array a end of useful data char
        histogram[histogram_length + 1] = '$';
        putchar('\n');
        int column_index = 0;
        int line_index = 0;
    
        // Print horizontal histogram
        printf("Horizontal Histogram\n--------------------\n");
        while (histogram[column_index] != '$'){
            printf("%3d: \t", column_index + 1);
            for (
                line_index = 0; 
                line_index < histogram[column_index]; 
                ++line_index) { 
                putchar('#'); 
            }
            putchar('\n');
            ++column_index;
        }
        // TBC

#### Lesson 13C: Histogram of Lengths of Input Words (Vertical)

        putchar('\n');
        // Print a vertical histogram
        printf("Vertical Histogram\n------------------\n");
        for (line_index = max_word_count; line_index >= 0; --line_index){
            column_index = 0;
            while (histogram[column_index] != '$'){
                if (line_index == 0){ printf("%2d ", column_index + 1); } 
                else if (histogram[column_index] >= line_index) { 
                    printf("## "); 
                } 
                else{ printf("   "); }
                ++column_index;
            }
            putchar('\n');
        }
        return 0;
    }

#### Lesson 14: Histogram of Freq of Different Characters

    // Write a program to print a histogram of the frequencies of different 
    // characters in its input.
    #include <stdio.h>
    #define ALPHA_NR 26
    #define NUM_NR 10

    int main(void){
        int i;
        char chars_freq[ALPHA_NR + NUM_NR];
        // Initialize the chars_freq array with 0
        for (i = 0; i < (ALPHA_NR + NUM_NR); ++i) { chars_freq[i] = 0; }
        // Count characters from the standard input
        char c;
        while ((c = getchar()) != EOF) {
            if (c >= 'a' && c <= 'z') { ++chars_freq[c - 'a']; } 
            else if (c >= '0' && c <= '9') { 
                ++chars_freq[c - '0' + ALPHA_NR]; 
            }
        }
        // Print horizontal histogram
        for (i = 0; i < (ALPHA_NR + NUM_NR); ++i){
            if (i < ALPHA_NR){ printf("%c: ", 'a' + i); } 
            else if (i >= ALPHA_NR){ printf("%c: ", '0' + i - ALPHA_NR); }
            int j;
            for (j = 0; j < chars_freq[i]; ++j){ printf("#"); }
            putchar('\n');
        }
        return 0;
    }

#### Lesson 15: Functions

    // Rewrite the temperature conversion program of Section 1.2 to use a 
    // function for conversion.
    #include <stdio.h>
    int main(void) {
        float celsius, fahr;
        int lower, upper, step;
        lower = 0;
        upper = 300;
        step = 30;
        // Printing a heading abouve the table
        printf("Celsius\t\tFahr.\n");
        printf("----------------------\n");
        celsius = lower;
        while (celsius <= upper) {
            fahr = celsius_to_fahrenheit(celsius);
            printf("%3.0f\t\t%6.1f\n", celsius, fahr);
            celsius += step;
        }
        return 0;
    }

    float celsius_to_fahrenheit(int celsius) { 
        return (9.0 / 5.0) * celsius + 32.0f; 
    }

#### Lesson 16A: Longest Line

    // Revise the main routine of the longest-line program so it will correctly
    // print the length of arbitrary long input lines, and as much as possible 
    // of the text.
    #include <stdio.h>
    #define MAXLINE 1000
    int get_line(char line[], int maxline);
    void copy(char to[], char from[]);

    int main(void){
        int len;
        int max = 0;
        char line[MAXLINE];
        char longest[MAXLINE];
        while ((len = get_line(line, MAXLINE)) > 0) {
            if (len > max) { max = len; copy(longest, line); }
        }
        if (max > 0) { printf("Longest line length: %d\n%s", max, longest); }
        return 0;
    }

#### Lesson 16B: Longest Line

    int get_line(char s[], int lim){
        int c, i;

        for (i = 0; i < lim - 1 && (c = getchar()) != EOF && c != '\n'; ++i) {
            s[i] = c; 
        }
        if (c == '\n') { s[i] = c; ++i; }
        s[i] = '\0';
        if (c != EOF && c != '\n') {
            while ((c = getchar()) != EOF && c != '\n') { ++i; }
            if (c == '\n') { ++i; }
        }
        return i;
    }

    void copy(char to[], char from[]) {
        int i = 0;
        while ((to[i] = from[i]) != '\0') { ++i; }
    }

#### Lesson 17: Input Lines > 80 Chars

    // Write a program to print all input lines that are longer than 80 
    // characters.
    #include <stdio.h>
    #define MAXLINE 1000
    #define LIMIT 80
    int get_line(char line[], int max_line_len);

    int main(void) {
        int len;
        char line[MAXLINE];
        while ((len = get_line(line, MAXLINE)) > 0) {
            if (len > LIMIT) { printf("%s", line); }
        }
        return 0;
    }

    int get_line(char line[], int max_line_len) {
        int c, i;
        for (
            i = 0; 
            i < max_line_len - 1 && (c = getchar()) != EOF && c != '\n'; 
            ++i) { 
            line[i] = c; 
        }
        if (c == '\n') { line[i] = c; ++i; }
        line[i] = '\0';
        return i;
    }

#### Lesson 18: Cleaning Input

    // Write a program to remove trailing blanks and tabs from each line of 
    // input, and to delete entirely blank lines.
    #include <stdio.h>
    #define MAXLINE 1000
    int get_line(char line[], int max_line_len);
    void remove_trailing_blanks(char line[], int length);

    int main(void){
        int len;
        char line[MAXLINE];
        while ((len = get_line(line, MAXLINE)) > 0){
            remove_trailing_blanks(line, len);
            printf("%s", line);
        }
        return 0;
    }

    int get_line(char line[], int max_line_len){
        int c, i;
        for (
            i = 0; 
            i < max_line_len - 1 && (c = getchar()) != EOF && c != '\n'; 
            ++i){ 
            line[i] = c; 
        }
        if (c == '\n') { line[i] = c; ++i; }
        line[i] = '\0';
        return i;
    }

    void remove_trailing_blanks(char line[], int length){
        int i;
        for (i = length - 2; line[i] == ' ' || line[i] == '\t'; --i);
        line[i + 1] = '\n';
        line[i + 2] = '\0';
    }

#### Lesson 19A: Reverse String

    // Write a function reverse(s) that reverses the character string s. Use 
    // it to write a program that reverses its input a line at a time.
    #include <stdio.h>
    #define MAXLINE 1000
    int get_line(char line[], int max_line_len);
    int length(char line[]);
    void reverse(char line[]);

    int main(void){
        int len;
        char line[MAXLINE];
        while ((len = get_line(line, MAXLINE)) > 0) { 
            reverse(line); printf("%s", line); 
        }
        return 0;
    }

    int get_line(char line[], int max_line_len){
        int c, i;
        i = 0;
        while (i < max_line_len - 1 && (c = getchar()) != EOF && c != '\n') { 
            line[i] = c; ++i; 
        }
        // flush out input stream if exceeding max_line_len limit
        while (i >= max_line_len - 1 && (c = getchar()) != '\n');
        if (c == '\n') { line[i] = '\n'; ++i; }
        line[i] = '\0';
        return i;
    }
    // TBC

#### Lesson 19B: Reverse String

    int length(char line[]){
        int i;
        for (i = 0; line[i] != '\0'; ++i);
        return i;
    }

    void reverse(char line[]){
        int i_front = 0;
        int i_back = length(line);
        char temp;
        if(line[i_back - 1] == '\n') { i_back -= 2; }
        else { i_back -= 1; }
        while (i_back > i_front) {
          temp = line[i_front];
          line[i_front] = line[i_back];
          line[i_back] = temp;
          ++i_front;
          --i_back;
        }
    }

### Section C: Chapter 1 Exercises 20-24

#### Lesson 20: detab

    // Write a program detab that replaces tabs in the input with the proper 
    // number of blanks to space to the next tab stop. Assume a fixed set of 
    // tab stops, say every n columns. Should n be a variable or a symbolic 
    // parameter? 
    #include <stdio.h>
    #define TABINC 8

    int main() {
        int c;
        int pos = 0;

        while ((c = getchar()) != EOF) {
            if (c == '\t') {
                int spaces = TABINC - (pos % TABINC);
                while (spaces > 0) { putchar(' '); pos++; spaces--; }
            } else if (c == '\n') { putchar(c); pos = 0;
            } else { putchar(c); pos++; }
        }
        return 0;
    }

#### Lesson 21: entab

    // Write a program entab that replaces strings of blanks by the minimum 
    // number of tabs and blanks to achieve the same spacing. Use the same tab
    // stops as for detab. When either a tab or a single blank would suffice 
    // to reach a tab stop, which should be given preference?
    #include <stdio.h>
    #define TAB_LENGTH 8

    int main(void) {
        int c;
        unsigned int line_pos = 0;
        unsigned int nr_of_spaces = 0;
        while ((c = getchar()) != EOF) {
            ++line_pos;
            if (c == ' '){
                ++nr_of_spaces;
                if (line_pos % TAB_LENGTH == 0 && nr_of_spaces > 1) { 
                    putchar('\t'); nr_of_spaces = 0; 
                }
            } else{
                while(nr_of_spaces) { putchar(' '); --nr_of_spaces; }
                if (c == '\n'){ line_pos = 0; }
                putchar(c);
            }
        }
        return 0;
    }

#### Lesson 22A: Fold Long Input Lines

    // Write a program to "fold" long input lines into two or more shorter 
    // lines after the last non-blank character that occurs before the n-th 
    // column of input. Make sure your program does something intelligent with 
    // very long lines, and if there are no blanks or tabs before the specified
    // column.
    #include <stdio.h>
    #define MAXLINE 10000
    #define TRUE (1 == 1)
    #define FALSE !TRUE
    #define BREAKING_POINT 40
    #define OFFSET 10
    int get_line(char line[], int max_line_len);
    void fold_line(char line[], char fold_str[], int n_break);

    int main(void){
        char line[MAXLINE];
        char fold_str[MAXLINE];
        while ((get_line(line, MAXLINE)) > 0){
            fold_line(line, fold_str, BREAKING_POINT);
            printf("%s", fold_str);
        }
        return 0;
    }

    int get_line(char line[], int max_line_len){
        int c, i = 0;
        while (i < max_line_len - 1 && (c = getchar()) != EOF && c != '\n'){ 
            line[i++] = c; 
        }
        if (c == '\n'){ line[i++] = c; }
        line[i] = '\0';
        return i;
    }

#### Lesson 22B: Fold Long Input Lines

    void fold_line(char line[], char fold_str[], int n_break){
        int i, j;
        int column = 0;
        int split = FALSE;
        int last_blank = 0;
        for (i = 0, j = 0; line[i] != '\0'; ++i, ++j){
            fold_str[j] = line[i];
            if (fold_str[j] == '\n'){ column = 0; }
            column++;
            if (column == n_break - OFFSET){ split = TRUE; }
            if (split && (fold_str[j] == ' ' || fold_str[j] == '\t')){ 
                last_blank = j; 
            }
            if (column == n_break){
                if (last_blank){ 
                    fold_str[last_blank] = '\n'; 
                    column = j - last_blank; 
                    last_blank = 0; 
                }
                else { fold_str[j++] = '-'; fold_str[j] = '\n'; column = 0; }
                split = FALSE;
            }
        }
        fold_str[j] = '\0';
    }

#### Lesson 23A: Remove Comments

    // Write a program to remove all comments from a C program. Don't forget to
    // handle quoted strings and character constants properly. C comments don't 
    // nest.
    // NOTE: This executes via `./a.out < main.c > out.c`
    #include <stdio.h>
    #define MAXSTR 10000
    #define TRUE (1 == 1)
    #define FALSE !TRUE
    // This is a test comment.
    int get_str(char str[], int limit); // This is another test comment.
    void remove_comments(char str[], char no_com_str[]);

    int main(void) {
        /**
         * This is multiline
         * block
         * comment.
        */

        char str[MAXSTR];
        char no_com_str[MAXSTR];
        get_str(str, MAXSTR);
        remove_comments(str, no_com_str);
        printf("%s", no_com_str);
        return 0;
    }

    int get_str(char str[], int limit) {
        int c, i = 0;
        while (i < limit - 1 && (c = getchar()) != EOF) { str[i++] = c; }
        str[i] = '\0';
        return i;
    }

#### Lesson 23B: Remove Comments

    void remove_comments(char str[], char no_com_str[]) {
        int in_quote = FALSE;
        int line_comment = FALSE;
        int block_comment = FALSE;

        int i = 0, j = 0;
        while (str[i] != '\0'){
            if (!block_comment){
                if (!in_quote && str[i] == '"') { in_quote = TRUE; } 
                else if (in_quote && str[i] == '"') { in_quote = FALSE; }
            }

            if (!in_quote) {
                if (str[i] == '/' && str[i + 1] == '*' && !line_comment) { 
                    block_comment = TRUE; 
                }
                if (str[i] == '*' && str[i + 1] == '/') { 
                    block_comment = FALSE; i += 2; 
                }
                if (str[i] == '/' && str[i + 1] == '/') { 
                    line_comment = TRUE; 
                }
                if (str[i] == '\n') { line_comment = FALSE; }
                if (line_comment || block_comment) { ++i; }
                else if (!line_comment || !block_comment) { 
                    no_com_str[j++] = str[i++]; 
                }
            } else { no_com_str[j++] = str[i++]; }
        }
        no_com_str[j] = '\0';
    }

#### Lesson 24A: Syntax Errors

    // Write a program to check a C program for rudimentary syntax errors like
    // unmatched parentheses, brackets and braces. Don't forget about quotes, 
    // both single and double, escape sequences, and comments. This program is
    // hard if you do it in full generality.
    // NOTE: This runs via `./a.out < test.c`
    #include <stdio.h>
    #define MAXSTR 10000
    #define TRUE (1 == 1)
    #define FALSE !TRUE
    int get_str(char str[], int limit);
    void check_syntax(char str[]);

    int main(void) {
        char str[MAXSTR];
        get_str(str, MAXSTR);
        check_syntax(str);
        return 0;
    }

    int get_str(char str[], int limit) {
        int c, i = 0;
        while (i < limit - 1 && (c = getchar()) != EOF) { str[i++] = c; }
        str[i] = '\0';
        return i;
    }

#### Lesson 24B: Syntax Errors

    void check_syntax(char str[]) {
        int parentheses = 0;
        int brackets = 0;
        int braces = 0;
        int single_quotes = FALSE;
        int double_quotes = FALSE;
        int block_comment = FALSE;
        int line_comment = FALSE;
        int i = 0;
        while ( str[i] != '\0' && parentheses >= 0 && brackets >= 0 
            && braces >= 0) {
            if (!line_comment && !block_comment && !single_quotes 
                && !double_quotes) {
                if (str[i] == '(') { ++parentheses; }
                else if (str[i] == ')') { --parentheses; }
                if (str[i] == '[') { ++brackets; }
                else if (str[i] == ']') { --brackets; }
                if (str[i] == '{') { ++braces; }
                else if (str[i] == '}') { --braces; }
            }
            if (!line_comment && !block_comment) {
                if (str[i] == '\'' && !single_quotes && !double_quotes) { 
                    single_quotes = TRUE; } 
                else if (single_quotes && str[i] == '\'' && (str[i - 1] != '\\' 
                    || str[i - 2] == '\\')) { single_quotes = FALSE; }
                if (str[i] == '"' && !single_quotes && !double_quotes) { 
                    double_quotes = TRUE; }
                else if (double_quotes && str[i] == '"' && 
                    (str[i - 1] != '\\' || str[i - 2] == '\\')) { 
                    double_quotes = FALSE; }
            }
            // TBC

#### Lesson 24C: Syntax Errors

            if (!single_quotes && !double_quotes){
                if (str[i] == '/' && str[i + 1] == '*' && !line_comment) { 
                    block_comment = TRUE; }
                else if (str[i] == '*' && str[i + 1] == '/') { 
                    block_comment = FALSE; }
                if (str[i] == '/' && str[i + 1] == '/' && !block_comment) { 
                    line_comment = TRUE; }
                else if (str[i] == '\n') { line_comment = FALSE; }
            }
            ++i;
        }

        if (parentheses) { printf("Error: unbalanced parentheses.\n"); }
        if (brackets) { printf("Error: unbalanced brackets.\n"); }
        if (braces) { printf("Error: unbalanced braces.\n"); }
        if (single_quotes) { printf("Error: unbalanced single quotes.\n"); }
        if (double_quotes) { printf("Error: unbalanced double quotes.\n"); }
        if (block_comment) { printf("Error: block comment not closed.\n"); }
    }
