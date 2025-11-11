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

#### Lesson 2A: Top 10 Things to Inspect the First Time You Access a Dataframe (1-5) 

    # 1. Columns, Data types, schema, and sampling
    df.columns
    df.dtypes
    df.info()
    df.shape
    df.head()
    df.sample(5)
    df.sample().T # transpose any random row
    df.tail()
    print(df.to_string()) # prints all rows in a df, useful for printing
                          # grouped dfs with more than 10 rows
    df['col_name'].nunique() # get count of unique values of a column
    df['col_name'].unique() # get list of unique values of a column

    # 2. Duplicate rows & subset
    df.duplicated().sum()
    df.duplicated(subset=['id', 'date']).sum()

    # 3. Missing values
    df.isnull().sum()
    df.isnull().mean() * 100  # % missing
    df = df[df['datetime_col'].notna()] # Filter out rows with certain missing values

    # 4. Primary key
    df.set_index(['col1','col2'], verify_integrity=True) 
    # The above will throw integrity error if the set is not a primary key. In
    # case of error, either change the set, or do: 
    df.drop_duplicates(subset=['col1','col2']) 

    # 5. Frequency of unique values across a column/ set of columns
    df.groupby('plan_id').size() # same logic as df.value_counts(), both return Series
    df.groupby(['plan_id','mac']).size()
    # Alternatively:
    df['plan_id'].value_counts().sort_index()

#### Lesson 2B: Top 10 Things to Inspect the First Time You Access a Dataframe (6-10) 

    # 6. Summary stats - look for impossible values (e.g., negative age),
    # extreme outliers, or unexpected categories. Gives: count, unique, mean, freq, 
    # top (mode), std, min, max, quantiles
    df.describe(include='all')
    df.describe(include='all').loc['count'].T # deep dive aesthetically

    # 7. Outliers & Anomalies (IQR method)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))
    outliers = outlier_condition.sum()
    # Filter out outliers
    non_outlier_mask = ~outlier_condition.any(axis=1)
    df_clean = df[non_outlier_mask]

    # 8. Correlations & Multicollinearity
    corr_matrix = df.corr(numeric_only=True)  

    # 9. Domain Consistency & Business Logic Checks
    assert (df['age'] >= 0).all(), 'Negative ages found!'

    # 10. Quick filteration / masking based analysis
    row_condition = df['assigned'].notna()
    df[['mobile', 'account_id', 'assigned', 'otp']][row_condition]

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

    # Rename specific columns 
    df = df.rename(columns={'old_name1': 'new_name1', 'old_name2': 'new_name2'})

	# Lowercase all column names
	df.columns = df.columns.str.lower()

    # Sort by column
    df.sort_values(by='sum_cnts',ascending=False)

    # Filter out rows where a specific col has null values
    df = df[df['datetime_col'].notna()] 

    # Filtering rows and columns in one line
    df[['mobile', 'account_id', 'assigned', 'otp']][df['assigned'].notna()]

#### Lesson 3: Filtering 

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

    # Filtering rows and columns in one line
    df[['mobile', 'account_id', 'assigned', 'otp']][df['assigned'].notna()]

#### Lesson 4: Using Python to Implement the Relational Model

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

#### Lesson 5: Indexing Advantages 

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

#### Lesson 7A: groupby (unindexed dataframe) 

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

#### Lesson 7B: groupby (indexed dataframe) 

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

#### Lesson 7C: groupby (shorthand syntax) 

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

#### Lesson 8A: Feature Engineering (Creating Helper Columns)

    # 1. Append quantile bin classification column
    df['util_range_qbc'] = pd.qcut(df['utilisation'],q=10,duplicates='drop',labels=False)
	df['util_range_qbc'].value_counts().sort_index()
    # NOTE: pd.qcut gives quantile / equal-frequency bins cut by data 
    # quantiles so bins have ~equal counts. Here, we drop duplicates to
    # merge duplicate bins caused by too many duplicate values  

    # 2. Appwnd bin classifcation column
	df['days_rng_bc'] = pd.cut(df['number_days'], bins=[0, 10, 20, 28, 35, float('inf')], labels=False) 
	df['days_rng_bc'].value_counts().sort_index()
    # NOTE: labels=False gives us the index number of the label (which can
    # directly be used as a numeric feature), instead of the label itself. Don't
    # add this param if you want the col to be more human readable, instead.

    # NOTE: It is good practice to use _bc and _qbc as indicators for 'bin
    # classification' and 'quantile bin classification', respectively

    # 3. Append cohort column
    conditions = [
        (df['id'].notnull() & df['otp'].isnull()),
        (df['id'].isnull() & df['otp'].notnull()),
        (df['id'].notnull() & df['otp'].notnull()),
        (df['id'].isnull() & df['otp'].isnull()),
    ]
    choices = ['CALL_NOINSTALL', 'NOCALL_INSTALL', 'CALL_INSTALL', 'NOCALL_NOINSTALL']
    df['cohort'] = np.select(conditions, choices, default=None)

#### Lesson 8B: Feature Engineering (Creating Helper Columns)

    # 4. Append computation storage column
	df['utilisation'] = df['number_days'] / df['plan_duration']

    # 5. Append boolean attribute columns
	df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0) 
	df['mac_80%'] = np.where((df['utilisation'] > 0.8) & (df['utilisation'] <= 0.9), 1, 0)

#### Lesson 9A: Pivot (single index and multi index)

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

#### Lesson 9B: Pivot (`pivot_table`)

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

#### Lesson 9C: Pivot (flattening a multi index)

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

#### Lesson 9D: Pivot Table (Motivation x Ability Grid)

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

#### Lesson 10A: Subset Aggregation / SQL sub-queries (using groupby)

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

    # Get min OTP_VERIFIED time per mobile and account_id )
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


#### Lesson 10B: Subset Aggregation / SQL sub-queries (using pivot)

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
