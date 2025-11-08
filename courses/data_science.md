# Data Science

## Part I: Replace Excel with Python

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
    df = df.set_index(['id','model'])
    df.index
    #! MultiIndex([(7937748,              '23090RA98I'),
    #!             (7938077,    'motorola edge 50 neo'),
    #!             ...
    #!             (7839768,                   'V2307')],
    #!            names=['id', 'model'], length=356540)
    df.index.names
    #! FrozenList(['id','model'])

#### Lesson 2A: Top 10 Things to Inspect the First Time You Access a Dataframe (1-5) 

    # 1. Columns, Data types, schema, and sampling
    df.columns
    df.dtypes
    df.info()
    df.shape
    df.head()
    df.sample(5)
    df.tail()

    # 2. Duplicate rows & subset
    df.duplicated().sum()
    df.duplicated(subset=['id', 'date']).sum()

    # 3. Unique value counts 
    for col in df.columns:
        uniques = df[col].unique()
        print(f'{col}: {uniques}')

    # 3. Candidate keys
    candidates = [col for col in df.columns if df[col].nunique() == len(df)]
    print('Candidates:', candidates)

    # 5. Missing values
    df.isnull().sum()
    df.isnull().mean() * 100  # % missing

#### Lesson 2B: Top 10 Things to Inspect the First Time You Access a Dataframe (6-10) 

    # 6. Summary stats - look for impossible values (e.g., negative age),
    # extreme outliers, or unexpected categories.
    df.describe(include='all')

    # 7. Value distributions
    df['category_col'].value_counts().sort_index()

    # 8. Outliers & Anomalies (IQR method)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum()

    # 9. Correlations & Multicollinearity
    corr_matrix = df.corr(numeric_only=True)  

    # 10. Domain Consistency & Business Logic Checks
    assert (df['age'] >= 0).all(), 'Negative ages found!'

#### Lesson 2: Modifications / Cleaning Based on Initial Inspection 

    df.info()

    # Convert int parseable object column to int64 (not int32 because it is non-nullable)
    df['android_version'] = pd.to_numeric(df['android_version'], errors='coerce').astype('Int64')

    # Convert float parseable object column to float64
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce').astype('float64')

    # Convert datetime parseable string columns to datetime
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')  

    # Convert datetime parseable int64 unix column to datetime
    df['unix'] = pd.to_datetime(df['unix'], unit='ms')

	# Lowercase all column names
	df.columns = df.columns.str.lower()

    # Sort by column
    df.sort_values(by='sum_cnts',ascending=False)

    # Filter out rows where a specific col has null values
    df = df[df['datetime_col'].notna()] 

#### Lesson 3: Filtering by Columns

    # Boolean filter
    df[((df['plan_duration'] > 12) & (df['status'].isin([6,12,24]))) | (df['plan_type'] == 'promo')]

    # Exclusion
    df[~df['plan_id'].isin([4, 5])]

    # isna and notna
    df[df['mac'].isna()]
    df[df['mac'].notna()]

    # Mask example
    mask = (df['plan_duration'] > 12) & (df['plan_id'] == 3)
    df.loc[mask, ['mac', 'mobile', 'plan_id']]
    df.loc[mask, 'plan_duration'] = 0

    # String filters
    df[df['mobile'].str.contains('555', na=False)]
    df[df['mac'].str.startswith('aa', na=False)]

    # Mutate/ Copy
    df = df[df['plan_duration'] > 12]
    filtered_df = df[df['plan_duration'] > 12].copy()

    # Range comparisons / between - readable and vectorized
    df[df['ts'] >= pd.Timestamp('2020-01-01')] # single-side
    df[df['ts'].between('2020-01-01', '2020-01-31')] # inclusive range (clean)

    # Component masks with .dt - year/month/weekday/time
    df[df['ts'].dt.year == 2020] # filter by year
    df[df['ts'].dt.month.isin([1,2,3])] # filter months
    df[df['ts'].dt.weekday < 5] # weekday mask (Mon=0)
    df[df['ts'].dt.time.between(pd.to_datetime('08:00').time(), pd.to_datetime('17:00').time())] # time-only mask

#### Lesson 4A: Using Python to Implement the Relational Model

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
    df.drop_duplicates()
    df.set_index('id')
    n = len(df.columns)
    columns = df.columns

#### Lesson 4B: Using Pyhton to Implement the Relational Model (set and reset index)

    #! data = {
    #!     'employee_id': [101, 102, 101, 103],
    #!     'department': ['HR', 'Engineering', 'Engineering', 'Sales'],
    #!     'name': ['Alice', 'Bob', 'Charlie', 'David'],
    #!     'salary': [60000, 80000, 75000, 70000]
    #! }
    df = pd.DataFrame(data)

    # Before setting index: It's just a pile of rows
    print(df)
    #!    employee_id   department     name  salary
    #! 0          101           HR    Alice   60000
    #! 1          102  Engineering      Bob   80000
    #! 2          101  Engineering  Charlie   75000
    #! 3          103        Sales    David   70000

    # Set the primary key columns as a MultiIndex. Use verify_integrity to 
    # catch duplicates. If there are duplicates, it'll yell at you, because 
    # primary keys shouldn't repeat.
    df.set_index(['employee_id', 'department'], inplace=True, verify_integrity=True)
    #!                             name  salary
    #! employee_id department
    #! 101         HR             Alice   60000
    #! 102         Engineering      Bob   80000
    #! 101         Engineering  Charlie   75000
    #! 103         Sales          David   70000

    # Go back to the default integer index
    df.reset_index(inplace=True)

#### Lesson 5: Filtering by Index 

    #!    employee_id   department  hire_date     name  salary
    #! 0          101           HR 2023-01-01    Alice   60000
    #! 1          102  Engineering 2023-01-04      Bob   80000
    #! 2          101  Engineering 2023-01-02  Charlie   75000
    #! 3          103        Sales 2023-01-03    David   70000

    df = df.set_index(['employee_id', 'department', 'hire_date'])
    df = df.sort_index()  # Sorts rows in the order of the index 
    #!                                        name  salary
    #! employee_id department  hire_date
    #! 101         Engineering 2023-01-02  Charlie   75000
    #!             HR          2023-01-01    Alice   60000
    #! 102         Engineering 2023-01-04      Bob   80000
    #! 103         Sales       2023-01-03    David   70000

    # 1. Fast lookups. Grab the row for employee 101 in Engineering on 2023-01-02
    df.loc[(101, 'Engineering', '2023-01-02')]
    #! name      Charlie
    #! salary      75000

    # 2. Datetime index slicing by temporarily setting hire_date as the single index
    temp_df = df.reset_index().set_index('hire_date').sort_index().loc['2023-01-01':'2023-01-03']
    #!             employee_id   department     name  salary
    #! hire_date
    #! 2023-01-01          101           HR    Alice   60000
    #! 2023-01-02          101  Engineering  Charlie   75000
    #! 2023-01-03          103        Sales    David   70000

    # 3. Partial string slicing (e.g., all of January 2023)
    temp_df.loc['2023-01']
    #!             employee_id   department     name  salary
    #! hire_date
    #! 2023-01-01          101           HR    Alice   60000
    #! 2023-01-02          101  Engineering  Charlie   75000

#### Lesson 6A: Joins (union join aka full outer join)

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

#### Lesson 6B: Joins (left join)

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

#### Lesson 6C: Joins (inner join)

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

#### Lesson 7: Group 

    # Group by column
    df = df.groupby(['plan_id', 'mobile']).agg(
        nmbr_mac=('mac', 'count'),
        # other aggs, if any
    )
    # NOTE: 
    # - Available aggs: count, nunique, min, max, first, last, sum, mean,
    #   median, mode

    # If you have just a single agg to do, you can shorten the syntax like this
    df.groupby(level=['plan_id', 'mobile']).mean(numeric_only=True)

    # Group by index
    df = df.set_index(['plan_id', 'mobile'])
    df.groupby(level=['plan_id', 'mobile']).mean(numeric_only=True)

#### Lesson 8: Feature Engineering (Creating Helper Columns)

    # Feature engineering: Discretizing continuous variable columns into bins
	df['days_rng_bc'] = pd.cut(df['number_days'], bins=[0, 10, 20, 28, 35, float('inf')], labels=False) 
	df['days_rng_bc'].value_counts().sort_index()
    # NOTE: labels=False gives us the index number of the label (which can
    # directly be used as a numeric feature), instead of the label itself. Don't
    # add this param if you want the col to be more human readable, instead.

    # Append additional derived continuous variable columns
	df['utilisation'] = df['number_days'] / df['plan_duration']

    # Append boolean attribute columns
	df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0) 
	df['mac_80%'] = np.where((df['utilisation'] > 0.8) & (df['utilisation'] <= 0.9), 1, 0)

    # Append quantile bin classification column
    df['util_range_qbc'] = pd.qcut(df['utilisation'],q=10,duplicates='drop',labels=False)
	df['util_range_qbc'].value_counts().sort_index()
    # NOTE: pd.qcut gives quantile / equal-frequency bins cut by data 
    # quantiles so bins have ~equal counts. Here, we drop duplicates to
    # merge duplicate bins caused by too many duplicate values  

    # NOTE: It is good practice to use _bc and _qbc as indicators for 'bin
    # classification' and 'quantile bin classification', respectively

#### Lesson 9A: Pivot (single index and multi index)

    # We don't really need the .pivot and .pivot_table methods to pivot a dataframe. This is because, 
    # the below mathematical definition of a pivot table, makes it possible to pivot simply by unstacking a grouped aggregate
    # DEF: Given a relation (table) T with attributes {R_attrs} (row keys), {C_attrs} (column keys), and {V} (value(s)), and an aggregation function 
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

    single_index_pivot = df.groupby(['foo']).agg(baz_sum=('baz', 'sum'))
    print(single_index_pivot, single_index_pivot.columns)
    #!      baz_sum
    #! foo
    #! one        8
    #! two        7
    #! Index(['baz_sum'], dtype='object')

    multi_index_pivot = df.groupby(['foo','bar']).agg(baz_sum=('baz', 'sum'))
    print(multi_index_pivot, multi_index_pivot.columns)
    #!          baz_sum
    #! foo bar
    #! one A          6
    #!     B          2
    #! two A          3
    #!     B          4
    #! Index(['baz_sum'], dtype='object')

#### Lesson 9A: Pivot (flattening a multi index)

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

#### Lesson 9C: Pivot Table (Motivation x Ability Grid)

    # Goal: 3x3 table with cols: motivation, high_ability, med_ability, low_ability
    # Assume df has:
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
