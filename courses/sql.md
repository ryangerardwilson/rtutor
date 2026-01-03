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

## Part III: ANSI SQL

### Section 1: Data Science Workflow Patterns

#### Lesson 1: Introduction

    /* ANSI SQL is a standardized language promoting portability across database systems, 
    allowing queries to run with minimal changes on compliant DBMS. Modern tools like 
    MySQL, PostgreSQL, Snowflake, and Metabase align with it for interoperability, 
    maintainability, and ecosystem integration.

    However, for our focus on these tools in analytical workflows, we'll use the LIMIT 
    clause—a non-ANSI extension supported by all—for concise result limiting, 
    prioritizing usability over strict standards. Queries can be adapted to ANSI's 
    FETCH FIRST if needed.

    The following lessons provide practical SQL patterns for data science workflows.*/

#### Lesson 1: Inspections

    -- 1. Check if a specific column is a key candidate by itself
    SELECT COUNT(*) FROM your_table;
    SELECT COUNT(DISTINCT potential_candidate_key) FROM your_table;

    -- 2. Count all rows of each unique value of the source column
    SELECT source, COUNT(*) AS row_count FROM actions 
    GROUP BY source ORDER BY row_count DESC;

    -- 3. Count all rows of each unique combination of source and action_type
    SELECT source, action_type, COUNT(*) AS row_count FROM actions 
    GROUP BY source, action_type ORDER BY row_count DESC LIMIT 20;

    -- 4. Check for the presence and count of NULL values in a specific column
    SELECT COUNT(*) AS null_count FROM your_table WHERE your_column IS NULL;

    -- 5. Get basic statistics (min, max, average) for a numeric column
    SELECT MIN(numeric_column) AS min_value, MAX(numeric_column) AS max_value, AVG(numeric_column) AS avg_value FROM your_table;

    -- 6. Retrieve a small sample of rows for manual review
    SELECT * FROM your_table LIMIT 10;
    -- Note: LIMIT works across MySQL, PostgreSQL, Snowflake, and Metabase-connected databases for this purpose.

    -- 7. Find the most frequent values in a column (top N)
    SELECT your_column, COUNT(*) AS frequency FROM your_table
    GROUP BY your_column ORDER BY frequency DESC LIMIT 5;

    -- 8. Calculate the percentage of rows for each unique value in a column
    SELECT your_column, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM your_table) AS percentage
    FROM your_table GROUP BY your_column ORDER BY percentage DESC;

    -- 9. Check for duplicate rows based on multiple columns
    SELECT column1, column2, COUNT(*) AS dup_count FROM your_table
    GROUP BY column1, column2 HAVING COUNT(*) > 1 ORDER BY dup_count DESC;

    -- 10. Get the data range for a date column
    SELECT MIN(date_column) AS earliest_date, MAX(date_column) AS latest_date FROM your_table;

    -- 11. Count rows matching a specific condition
    SELECT COUNT(*) AS matching_rows FROM your_table WHERE your_column = 'specific_value';

    -- 12. List all unique values in a column (for small sets)
    SELECT DISTINCT your_column FROM your_table ORDER BY your_column;

#### Lesson 2: Joins

    -- 1. Join two tables to count matching rows based on a key
    SELECT COUNT(*) FROM table1 t1 JOIN table2 t2 ON t1.key_column = t2.key_column;

    -- 2. Get top N joined records with aggregation
    SELECT t1.category, SUM(t2.amount) AS total FROM table1 t1 JOIN table2 t2 ON t1.id = t2.table1_id GROUP BY t1.category ORDER BY total DESC LIMIT 5;

    -- 3. Check for unmatched (orphan) records in a left join
    SELECT COUNT(*) FROM table1 t1 LEFT JOIN table2 t2 ON t1.key = t2.key WHERE t2.key IS NULL;

    -- 4. Join and filter to list unique combinations
    SELECT DISTINCT t1.name, t2.type FROM table1 t1 JOIN table2 t2 ON t1.id = t2.table1_id WHERE t1.status = 'active' LIMIT 10;

    -- 5. Aggregate across joins for average per group
    SELECT t1.group_id, AVG(t2.value) AS avg_value FROM table1 t1 JOIN table2 t2 ON t1.id = t2.table1_id GROUP BY t1.group_id ORDER BY avg_value DESC;

    -- 6. Join three tables to count multi-level relations
    SELECT COUNT(*) FROM table1 t1 JOIN table2 t2 ON t1.id = t2.t1_id JOIN table3 t3 ON t2.id = t3.t2_id WHERE t3.condition = 'true';

#### Lesson 3: Partitions

    -- 1. Assign row numbers within partitions (e.g., per group)
    SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY date_column DESC) AS row_num FROM your_table LIMIT 20;

    -- 2. Calculate running total within partitions
    SELECT *, SUM(amount) OVER (PARTITION BY user_id ORDER BY date_column) AS running_total FROM transactions;

    -- 3. Rank values within partitions
    SELECT *, RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank FROM employees;

    -- 4. Get the first value in each partition
    SELECT *, FIRST_VALUE(value) OVER (PARTITION BY group_id ORDER BY date_column) AS first_in_group FROM your_table;

    -- 5. Compute average within partitions
    SELECT *, AVG(metric) OVER (PARTITION BY category) AS avg_per_category FROM your_table;

    -- 6. Lag (previous) value within partitions
    SELECT *, LAG(value) OVER (PARTITION BY user_id ORDER BY date_column) AS previous_value FROM your_table;

    -- 7. Lead (next) value within partitions
    SELECT *, LEAD(value) OVER (PARTITION BY user_id ORDER BY date_column) AS next_value FROM your_table;

    -- 8. Percent rank within partitions
    SELECT *, PERCENT_RANK() OVER (PARTITION BY group ORDER BY score DESC) AS percent_rank FROM scores LIMIT 10;

#### Lesson 4: JSON parsing

    -- 1. Extract a value from JSON using standard functions 
    SELECT JSON_VALUE(json_column, '$.key_name') AS extracted_value FROM your_table LIMIT 10;

    -- 2. Filter rows based on JSON value 
    SELECT * FROM your_table WHERE JSON_VALUE(json_column, '$.key_name') = 'desired_value' LIMIT 10;

    -- 3. Group and count by extracted JSON key 
    SELECT JSON_VALUE(json_column, '$.key_name') AS key_value, COUNT(*) AS count FROM your_table GROUP BY key_value ORDER BY count DESC;

    -- 4. Check if JSON contains a key 
    SELECT * FROM your_table WHERE JSON_EXISTS(json_column, '$.key_name') LIMIT 10;

    -- 5. Extract from nested JSON 
    SELECT JSON_VALUE(json_column, '$.nested_object.key_name') AS nested_value FROM your_table LIMIT 10;

    -- 6. Search for text within JSON string 
    SELECT * FROM your_table WHERE json_column LIKE '%"key_name":"desired_value"%' LIMIT 10;
    -- Note: This is a fallback for databases without native JSON functions; use with caution as it may match substrings incorrectly.

#### Lesson 5: Subqueries

    -- 1. Use a scalar subquery to compare against an aggregate
    SELECT name, salary FROM employees WHERE salary > (SELECT AVG(salary) FROM employees);

    -- 2. Subquery in SELECT for derived values
    SELECT name, (SELECT COUNT(*) FROM orders o WHERE o.employee_id = e.id) AS order_count FROM employees e LIMIT 10;

    -- 3. Correlated subquery for per-row comparisons
    SELECT e.name, e.department, (SELECT AVG(salary) FROM employees sub WHERE sub.department = e.department) AS dept_avg_salary FROM employees e;

    -- 4. Subquery in FROM clause (derived table)
    SELECT dept.department, dept.avg_salary FROM (SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department) AS dept WHERE dept.avg_salary > 50000;

    -- 5. EXISTS subquery to check for related records
    SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);

#### Lesson 6: CTEs

    -- 1. Basic CTE for query reuse
    WITH dept_averages AS (SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department)
    SELECT e.name, e.salary, da.avg_salary FROM employees e JOIN dept_averages da ON e.department = da.department WHERE e.salary > da.avg_salary LIMIT 10;

    -- 2. Recursive CTE for hierarchical data
    WITH RECURSIVE hierarchy AS (
        SELECT id, name, manager_id, 1 AS level FROM employees WHERE manager_id IS NULL
        UNION ALL
        SELECT e.id, e.name, e.manager_id, h.level + 1 FROM employees e JOIN hierarchy h ON e.manager_id = h.id
    )
    SELECT * FROM hierarchy ORDER BY level LIMIT 20;

#### Lesson 7: Pivots

    -- 1. Pivot data using CASE (rows to columns)
    SELECT product,
           SUM(CASE WHEN month = 'Jan' THEN sales ELSE 0 END) AS jan_sales,
           SUM(CASE WHEN month = 'Feb' THEN sales ELSE 0 END) AS feb_sales,
           SUM(CASE WHEN month = 'Mar' THEN sales ELSE 0 END) AS mar_sales
    FROM sales GROUP BY product ORDER BY product;

    -- 2. Unpivot data using UNION (columns to rows)
    SELECT product, 'Jan' AS month, jan_sales AS sales FROM pivoted_sales
    UNION ALL
    SELECT product, 'Feb' AS month, feb_sales AS sales FROM pivoted_sales
    UNION ALL
    SELECT product, 'Mar' AS month, mar_sales AS sales FROM pivoted_sales
    ORDER BY product, month;





