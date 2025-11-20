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

## Part II: Pretty SQL

### Section 1: SQLFluff Style

#### Lesson 1: SQLFluff config 

    This is the .sqlfluff config we add to our project root
    #! [sqlfluff]
    #! dialect = postgres
    #! exclude_rules = RF06  -- allow leading commas because trailing commas are for idiots

    #! [sqlfluff:rules:capitalisation.keywords]
    #! capitalisation_policy = upper

    #! [sqlfluff:rules:capitalisation.identifiers]
    #! capitalisation_policy = lower

    #! [sqlfluff:rules:capitalisation.functions]
    #! capitalisation_policy = upper

    #! [sqlfluff:rules:layout.comma]
    #! line_position = leading          -- git diff stays clean, fight me

#### Lesson 2: Keywords, Identifiers, Leading Commas 

    -- Keywords: Words the SQL parser reserves: SELECT, FROM, WHERE, JOIN, ON, LEFT, RIGHT, INNER, OUTER, GROUP BY, ORDER BY, AS, etc. Write them in UPPERCASE.
    -- Identifiers: Table names (users, posts), column names (id, username). 
    -- Aliases: The AS u / AS p bullshit. Shortens u.id instead of users.id. 
    -- NEVER use leading commas. This garbage is an abomination:

#### Lesson 3: Group By

    SELECT
        source,
        COUNT(*) AS action_count
    FROM actions
    GROUP BY source;

#### Lesson 4A: Joins (Inner, Left, Right)

    -- 1. INNER JOIN  
    -- This is the default behavior when you just write "JOIN".  
    -- It returns ONLY rows where there is a match in BOTH tables.  
    -- Users who have never written a post are completely excluded from the result set.  
    SELECT u.id, u.username, COUNT(p.id) AS total_posts
    FROM users AS u
    INNER JOIN posts AS p ON p.author_id = u.id
    GROUP BY u.id, u.username;
    -- Result: lonely users with zero posts do not appear at all. This is what your original
    -- broken query was doing, because you forgot the "LEFT".

    -- 2. LEFT JOIN (also called LEFT OUTER JOIN)  
    -- Returns ALL rows from the left table (users), and matching rows from the right table (posts).  
    -- If a user has no posts, the post columns will be NULL, but the user still shows up.  
    -- This is almost always what you want when counting things per user.  
    SELECT u.id, u.username, COUNT(p.id) AS total_posts  -- COUNT(p.id) ignores NULL rows!
    FROM users AS u
    LEFT JOIN posts AS p ON p.author_id = u.id
    GROUP BY u.id, u.username;
    -- Users with zero posts appear with total_posts = 0. Stop being an idiot and use this.

    -- 3. RIGHT JOIN (also called RIGHT OUTER JOIN)  
    -- The opposite: returns ALL rows from the right table (posts), and matching users from the left.  
    -- Useful when you suspect orphaned posts (e.g. author_id is NULL or points to a deleted user).  
    SELECT u.id, u.username, p.id AS post_id, p.title
    FROM users AS u
    RIGHT JOIN posts AS p ON p.author_id = u.id;
    -- Posts whose author no longer exists will show up with u.id = NULL and u.username = NULL.

#### Lesson 4B: Joins (Outer, Cross, Multiple)

    -- 4. FULL OUTER JOIN  
    -- Returns everything from both sides: all users AND all posts, matched where possible.  
    -- Rows without a match get NULLs on the opposite side.  
    -- PostgreSQL, SQLite (recent versions), and real databases support it. MySQL still cries in a corner.  
    SELECT COALESCE(u.username, '<< Orphaned post >>') AS display_name,
           COUNT(p.id) AS total_posts
    FROM users AS u
    FULL OUTER JOIN posts AS p ON p.author_id = u.id
    GROUP BY u.id, u.username;
    -- You'll see both lonely users (total_posts=0) and orphaned posts.

    -- 5. CROSS JOIN  
    -- Cartesian product: every row from the left table combined with every row from the right table.  
    -- No ON clause needed (or allowed). Useful to define parameters that we
    would frequently use across a very long query
    WITH 
    params AS (
        SELECT 
            '2025-11-18' AS start_day,
            '2025-11-25' AS end_day
    )
    SELECT user_id, created_at
    FROM users
    CROSS JOIN params p
    WHERE created_at BETWEEN p.start_day AND p.end_day;

    -- 6. Multiple joins in the same query  
    -- Real-world queries usually need more than one join.  
    -- Here we count both posts and comments per user, while still including users with nothing.  
    SELECT
        u.username,
        COUNT(DISTINCT p.id) AS posts,      -- DISTINCT because a post can have many comments
        COUNT(DISTINCT c.id) AS comments
    FROM users AS u
    LEFT JOIN posts AS p ON p.author_id = u.id
    LEFT JOIN comments AS c ON c.post_id = p.id   -- note: chained through posts
    GROUP BY u.id, u.username
    ORDER BY posts DESC, comments DESC;

#### Lesson 4C: Joins (Self)

    -- 7. Self-join - joining a table to itself  
    -- Classic example: hierarchical data like employees and their managers, or threaded comments.  
    -- Both sides are the same table, so we use aliases to keep our sanity.  
    -- who reports to whom (including the poor CEO who has no one to blame)
    SELECT
        e.username AS employee,
        COALESCE(m.username, '<< No manager (CEO or orphan) >>') AS manager
    FROM users AS e
    LEFT JOIN users AS m ON e.manager_id = m.id
    ORDER BY manager, employee;
    -- LEFT JOIN ensures people without a manager (top boss or data error) still appear.

#### Lesson 5A: CTEs - Treat them like Functions

    # Wrong (what every PHP monkey does):
    #! WITH
    #! tmp1 AS (SELECT * FROM users WHERE active),
    #! tmp2 AS (SELECT * FROM posts WHERE created_at > now() - interval '1 year'),
    #! tmp3 AS (SELECT author_id, count(*) FROM posts GROUP BY author_id)
    #! SELECT * FROM tmp1 JOIN tmp2 USING (user_id) JOIN tmp3 USING (author_id);
    # That shit is unreadable, untestable, and makes baby Codd cry.

#### Lesson 5B: CTEs - Treat them like Functions

    -- Correct - CTEs are FUNCTIONS, period:
    WITH
    active_users AS (                  -- function name: active_users()
        SELECT                         -- clear inputs: none
            user_id,                   -- clear outputs
            username
        FROM
            users
        WHERE
            active = true
    ),

    recent_posts AS (                  -- function name: recent_posts()
        SELECT                         -- returns one row per author
            author_id,
            COUNT(*) AS posts_written
        FROM
            posts
        WHERE
            created_at >= CURRENT_DATE - INTERVAL '1 year'
        GROUP BY
            author_id
    ),

    top_commenters AS (                -- another pure function
        SELECT
            author_id,
            COUNT(*) AS comments_written
        FROM
            comments
        WHERE
            created_at >= CURRENT_DATE - INTERVAL '1 year'
        GROUP BY
            author_id
    )

    SELECT
        u.username,
        COALESCE(p.posts_written, 0)    AS posts_written,
        COALESCE(c.comments_written, 0) AS comments_written
    FROM
        active_users AS u
        LEFT JOIN recent_posts    AS p ON p.author_id = u.user_id
        LEFT JOIN top_commenters  AS c ON c.author_id = u.user_id
    ORDER BY
        posts_written DESC,
        comments_written DESC
    LIMIT 50;


#### Lesson 5C: CTEs - Treat them like Functions

    1. Each CTE has a single responsibility - you can test it in isolation
    2. Clear name = self-documenting (no "what the fuck does tmp3 do?" at 3 a.m.)
    3. Inputs/outputs obvious - exactly like a function signature
    4. Reusable - other queries can reuse active_users() without copy-paste
    5. Optimizer loves it - Postgres materialises or inlines intelligently anyway
    6. If you break it, only one "function" breaks - not the entire 300-line monster

    Treat your CTEs like C functions. If your CTE is longer than 25 lines or does 
    three unrelated things, you split it into multiple CTEs like you would split a 
    500-line C function.
