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
    --! [sqlfluff]
    --! dialect = postgres
    --! exclude_rules = RF06  -- allow leading commas because trailing commas are for idiots

    --! [sqlfluff:rules:capitalisation.keywords]
    --! capitalisation_policy = upper

    --! [sqlfluff:rules:capitalisation.identifiers]
    --! capitalisation_policy = lower

    --! [sqlfluff:rules:capitalisation.functions]
    --! capitalisation_policy = upper

    --! [sqlfluff:rules:layout.comma]
    --! line_position = leading        

#### Lesson 2: Keywords, Identifiers, Leading Commas 

    -- Keywords:    Words the SQL parser reserves: SELECT, FROM, WHERE, JOIN, ON, LEFT, RIGHT, 
                    INNER, OUTER, GROUP BY, ORDER BY, AS, etc. Write them in UPPERCASE.
    -- Identifiers: Table names (users, posts), column names (id, username). 
    -- Aliases:     The AS u / AS p bullshit. Shortens u.id instead of users.id. 

#### Lesson 3A: Group By

    -- 1. Rows per single column
    SELECT
        source,
        COUNT(*) AS row_count
    FROM actions
    GROUP BY source
    ORDER BY row_count DESC;

    --! +----------+-----------+
    --! | source   | row_count |
    --! +----------+-----------+
    --! | web      |    842105 |
    --! | mobile   |    523441 |
    --! | api      |     98765 |
    --! | iframe   |      2341 |
    --! | bot      |       666 |
    --! +----------+-----------+

    -- 2. Two columns at once - instant "what the fuck is combining here?"
    SELECT
        source,
        action_type,
        COUNT(*) AS row_count
    FROM actions
    GROUP BY source, action_type
    ORDER BY row_count DESC
    LIMIT 20;

    --! +----------+-------------+-----------+
    --! | source   | action_type | row_count |
    --! +----------+-------------+-----------+
    --! | web      | click       |    412341 |
    --! | mobile   | view        |    298765 |
    --! | web      | purchase    |    123456 |
    --! | api      | login       |     87654 |
    --! | mobile   | add_to_cart |     65432 |
    --! ... (15 more rows)

#### Lesson 3B: Group By

    -- 3. Monthly distribution (Postgres/DuckDB/BigQuery/Trino version)
    SELECT
        DATE_TRUNC('month', created_at) AS month,
        COUNT(*) AS row_count
    FROM actions
    GROUP BY month
    ORDER BY month;

    --! +---------------------+-----------+
    --! | month               | row_count |
    --! +---------------------+-----------+
    --! | 2024-01-01 00:00:00 |     45231 |
    --! | 2024-02-01 00:00:00 |     48912 |
    --! | ...                 |           |
    --! | 2025-10-01 00:00:00 |    156789 |
    --! | 2025-11-01 00:00:00 |     98765 |   -- look, sudden spike, something happened
    --! +---------------------+-----------+

    -- 4. NULL / garbage detection - every table is lying to you
    SELECT
        CASE 
            WHEN user_id IS NULL THEN 'missing'
            WHEN user_id = 0 THEN 'dummy_zero'
            WHEN user_id < 0 THEN 'negative_wtf'
            ELSE 'valid'
        END AS user_id_quality,
        COUNT(*) AS row_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM actions
    GROUP BY user_id_quality
    ORDER BY row_count DESC;

    --! +-----------------+-----------+-------+
    --! | user_id_quality | row_count |  pct  |
    --! +-----------------+-----------+-------+
    --! | valid           |   1452341 | 98.45 |
    --! | missing         |     23456 |  1.59 |
    --! | dummy_zero      |       666 |  0.05 |
    --! +-----------------+-----------+-------+

#### Lesson 3C: Group By

    -- 5. Outlier users with HAVING - find the bots and the addicts
    SELECT
        user_id,
        COUNT(*) AS actions,
        MIN(created_at) AS first_seen,
        MAX(created_at) AS last_seen
    FROM actions
    GROUP BY user_id
    HAVING COUNT(*) > 10000 OR COUNT(*) = 1
    ORDER BY actions DESC
    LIMIT 30;

    --! +---------+---------+---------------------+---------------------+
    --! | user_id | actions | first_seen          | last_seen           |
    --! +---------+---------+---------------------+---------------------+
    --! | 666     |   87421 | 2024-01-15 03:14:11 | 2025-11-20 23:59:59 |  
    --! | 12345   |   54321 | 2024-06-01 12:00:00 | 2025-11-21 01:23:45 | 
    --! | 999999  |       1 | 2025-11-21 06:66:66 | 2025-11-21 06:66:66 |
    --! ... (lots of single-action rows)

    -- 6. Uniqueness check - is this column actually a key?
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT user_id) AS distinct_users,
        COUNT(DISTINCT session_id) AS distinct_sessions,
        ROUND(100.0 * COUNT(DISTINCT user_id) / COUNT(*), 4) AS pct_unique_user
    FROM actions;

    --! +------------+----------------+-------------------+-----------------+
    --! | total_rows | distinct_users | distinct_sessions | pct_unique_user |
    --! +------------+----------------+-------------------+-----------------+
    --! |    1478321 |         234567 |            987654 |         15.8674 |
    --! +------------+----------------+-------------------+-----------------+

#### Lesson 3D: Group By

    -- 7. Exact duplicate rows? (should be zero if you have a proper PK)
    SELECT
        source, action_type, user_id, created_at, metadata,
        COUNT(*) AS duplicate_count
    FROM actions
    GROUP BY source, action_type, user_id, created_at, metadata
    HAVING COUNT(*) > 1
    ORDER BY duplicate_count DESC;

    --! +--------+-------------+---------+---------------------+----------+-----------------+
    --! | source | action_type | user_id | created_at          | metadata | duplicate_count |
    --! +--------+-------------+---------+---------------------+----------+-----------------+
    --! | web    | click       |  12345  | 2025-11-21 10:11:12 | null     |              47 |  
    --! +--------+-------------+---------+---------------------+----------+-----------------+


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

### Section 2: Practical SQL Workflow Patterns

#### Lesson 1: Using Count and Group By for Quick Inspections

    -- 1. Count all records from a relations
    SELECT
        COUNT(*)
    FROM your_cte_or_table;

    -- 2. Count all rows of each unique value of the source column
    SELECT
        source,
        COUNT(*) AS row_count
    FROM actions
    GROUP BY source
    ORDER BY row_count DESC;

    -- 3. Count all rows of each unqiue combination of source and action_type
    SELECT
        source,
        action_type,
        COUNT(*) AS row_count
    FROM actions
    GROUP BY source, action_type
    ORDER BY row_count DESC
    LIMIT 20;


#### Lesson 2: Using Subset Aggregation for Cohort Analysis of Log Tables

    WITH base_data AS (
        /* Log tables are huge, so we start with a filter that is a little larger 
        than the cycle we want to isolate i.e. Oct 16 to Nov 16 */
        SELECT *
        FROM your_log_table
        WHERE added_time > '2025-10-09 00:00:00'
        AND added_time < '2025-11-23 00:00:00'
    ),
    cohort_base_identifier AS (
        /* Identifies cohort as cycles fully completed by November 16th when inner 
        joined with base_data */
        SELECT
            account_id,
            mobile
        FROM base_data
        GROUP BY account_id, mobile
        HAVING MAX(added_time) <= '2025-11-16 00:00:00'
    ),
    assignments AS (
        SELECT
            b.account_id,
            b.mobile,
            MIN(b.added_time) AS first_assigned
        FROM base_data b
        INNER JOIN cohort_base_identifier c
        ON b.account_id = c.account_id AND b.mobile = c.mobile
        WHERE b.event_name = 'ASSIGNED'
        GROUP BY b.account_id, b.mobile
        HAVING MIN(b.added_time) > '2025-10-16 00:00:00'
    ),
    verifications AS (
        SELECT
            b.account_id,
            b.mobile,
            MIN(b.added_time) AS otp_verified
        FROM base_data b
        INNER JOIN cohort_base_identifier c
        ON b.account_id = c.account_id AND b.mobile = c.mobile
        WHERE b.event_name = 'OTP_VERIFIED'
        GROUP BY b.account_id, b.mobile
    )
    /* Now, we simply do subset aggreagation */
    SELECT
        a.account_id,
        a.mobile,
        a.first_assigned,
        v.otp_verified
    FROM assignments a
    LEFT JOIN verifications v
    ON a.account_id = v.account_id AND a.mobile = v.mobile;
