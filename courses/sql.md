# SQL

## Part I: Relational Theory

### Section 1: EF Codd's Original Relational Model

#### Lesson 1: Context

    /* In a 1969 IBM Research Report, EF Codd proposed a relational model,
    while trying to solve the problem of "data dependence and pointer-chasing
    hell" in the 1960s hierachial databases:
    - data was duplicated
    - updates caused anomalies
    - data was accessed by navigating file paths, and ad hoc queries were 
      painful or impossible
    - there was no clean theoretical rooting for integrity or optimization

    The relational model provides a rigorous mathematical foundation
    (relations, algebra/calculus) for data science.

    The original model had three major components: 
    - structure 
    - integrity
    - manipulation */

#### Lesson 2A: Structural Features - Tabular Data as Mathematical Relations

    /* The principle strucutral feature is the idea that tabular data is merely
    a way to represent an n-ary mathematical relation, where 
    - n represents the number of columns, 
    - columns represent attributes of tuple indices, and 
    - rows represent tuples.

    Also,
    - 1 column = unary
    - 2 columns = binary
    - 3 columns = ternary
    - And so on. */

#### Lesson 2B: Structural Features - Keys 

    /*
    DEPT
    ----------------------------
    DNO | DNAME         | BUDGET
    ----------------------------
    D1  | Marketing     | 10M
    D2  | Development   | 12M
    D3  | Research      | 5M

    EMP: DEPT.DNO referenced by EMP.DNO
    --------------------------
    ENO | ENAME | DNO   | SALARY
    --------------------------
    E1  | Lopez | D1    | 40K
    E2  | Cheng | D1    | 42K
    E3  | Finzi | D2    | 30K
    E4  | Saito | D2    | 35K

    NOTE: We use {EL1, EL2 ...} syntax to denote a set, showing elements in a
    comma list.

    Keys/ Candidate Keys: {ATTRIBUTES} (i.e. a set of attributes), where each
    element is capable of uniquely identifying each tuple. Example: {DNO,DNAME}
    in DEPT, {ENO,ENAME} in EMP. 
    - Primary key: Is a specific ATTRIBUTE - if you have multiple keys, you
      pick one to be the main one. 
    - Foreign keys: {ATTRIBUTES}, where each element must match a key in
      another table. Example - {DNO} in EMP */
