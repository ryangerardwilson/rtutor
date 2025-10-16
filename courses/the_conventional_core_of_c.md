# The Conventional Core of C

## Lesson 1: Hello World

    #include "stdio.h"

    int main(){
        printf("Hello, world!");
        return 0;
    }

## Lesson 2: Compile & Execute

    cc main.c && ./a.out

## Lesson 3: Basic Types

    #include "stdio.h"

    int main(){
        int a = 5;
        char s[10] = "Linus";
        return 0;
    }

## Lesson 4: Basic Data Types and Their Qualifiers

    #include "stdio.h"

    int main() {

        // The 4 Complete Basic Data Types 
        // Compiler allocates memory based on type declaration.
        int i = 1337;              
        float f = 3.14f;           
        double d = 3.1415926535;   
        char c = 'L';             
        // Equivalent:
        char c2 = 76;

        return 0;
    } 

## Lesson 5: Derived Data Types

    #include "stdio.h"

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

## Lesson 6: Formatting with Printf

    #include "stdio.h"

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

## Lesson 7: Basic Arithmetic Expressions

    #include "stdio.h"

    int main() {
        double result;
        result = 2 * (2 + 3) / 4.0 - 1;
        printf("%f\n", result);
        return 0;
    } 

## Lesson 8: If-Else

    #include "stdio.h"

    int main() {
        int age = 25;
        if (age >= 18) {
            printf("Adult\n");
        } else {
            printf("Kid\n");
        }
        return 0;
    } 

## Lesson 9: While Loop

    #include "stdio.h"

    int main() {
        char s[] = "hello";
        int i = 0;
        while (s[i] != '\0') {
            i++;
        }
        printf("Length: %d\n", i);
        return 0;
    } 

## Lesson 10: For Loop - Repeat Without Being a Loop Idiot

    #include "stdio.h"

    int main() {
        int i;
        for (i = 0; i < 10; i++) {
            printf("%d\n", i * i);
        }
        return 0;
    } 

## Lesson 11: Getchar, Putchar, EOF

    #include "stdio.h"

    int main() {
        int c;
        printf("Type text, Ctrl+D to flush buffer"); 
        prrintf("OR: Enter, Ctrl+D to trigger EOT, which c interprets as EOF"); 

        // NOTE: EOF is
        // 1. not a character, but a macro defined in stdio.h as -1. 
        // 2. what getchar() spits back when there's no more input to read 
        while ((c = getchar()) != EOF) {
            putchar(c);
        }
        return 0;
    }

## Lesson 12: scanf

    #include "stdio.h"

    int main() {
        char name[20];
        printf("Enter name: ");
        scanf("%s", name);
        printf("Hello, %s\n", name);
        return 0;
    } 

## Lesson 13: Constants, Mutables, Globals & Functions

    #include "stdio.h"

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
