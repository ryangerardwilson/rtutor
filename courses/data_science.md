# Data Science

## Part I: Replace Excel with Python

### Section 1: Vanilla Numpy & Pandas

#### Lesson 1: Inspecting Dataframes (basics)

	# Get columns
	df.columns
	# Get tuple showing (row_count, col_count)
	df.shape
	# Preview first 5 rows
	df.head()

	# Get count of distinct non-null values
	df['mac'].nunique()
	df[['mac', 'plan_id', 'mobile']].nunique() # per-column counts excluding NaNs
    # Unique coung for all columns
    for c in df.columns: print(f'{c}: {df[c].nunique()}')

    # Get count of null/NaN values in a column
    df['col'].isna().sum()

    # Quick way to group by column
	df['plan_duration'].value_counts().sort_index() # Returns Pandas series 

	# Lowercase all column names
	df.columns = df.columns.str.lower()

    # Sort by column
    df.sort_values(by='sum_cnts',ascending=False)

#### Lesson 1B: Inspecting Dataframes (quickly identify candidate keys)

    for c in df.columns:
        unique_count_str = f'\nATTRIBUTE: {c} (unique : {df[c].nunique()})'
        if df[c].nunique() and df[c].nunique() < 10: 
            print(f"{unique_count_str}\nVAL CNTS:") 
            print(df[c].value_counts().sort_index()) 
            print()

#### Lesson 2: Filter & Mask (basics)


    # Boolean filter
    df[((df['plan_duration'] > 12) & (df['status'].isin([6,12,24]))) | (df['plan_type'] == 'promo')]

    # Exclusion
    df[~df['plan_id'].isin([4, 5])]

    # Mutate/ Copy
    df = df[df['plan_duration'] > 12]
    filtered_df = df[df['plan_duration'] > 12].copy()

    # Notna
    df[df['mac'].notna()]

    # Dropna
    df.dropna(subset=['mac', 'mobile'])

    # Mask example
    mask = (df['plan_duration'] > 12) & (df['plan_id'] == 3)
    df.loc[mask, ['mac', 'mobile', 'plan_id']]
    df.loc[mask, 'plan_duration'] = 0

    # String filters
    df[df['mobile'].str.contains('555', na=False)]
    df[df['mac'].str.startswith('aa', na=False)]

#### Lesson 2: Filter & Mask (datetime)

    # Ensure real datetimes - coerce bad input to NaT, then drop
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')  
    df['unix_ts_to_ts'] = pd.to_datetime(df['unix_ts_to_ts'], unit='ms')
    df = df[df['ts'].notna()] 

    # Range comparisons / between - readable and vectorized
    df[df['ts'] >= pd.Timestamp('2020-01-01')] # single-side
    df[df['ts'].between('2020-01-01', '2020-01-31')] # inclusive range (clean)

    # DatetimeIndex slicing - fastest and very readable
    df = df.set_index('ts') # set index once for time ops
    df.loc['2020-01-01':'2020-01-31'] # inclusive index slice
    df.between_time('08:00', '17:00') # time-of-day filter on index

    # Component masks with .dt - year/month/weekday/time
    df[df['ts'].dt.year == 2020] # filter by year
    df[df['ts'].dt.month.isin([1,2,3])] # filter months
    df[df['ts'].dt.weekday < 5] # weekday mask (Mon=0)
    df[df['ts'].dt.time.between(pd.to_datetime('08:00').time(), pd.to_datetime('17:00').time())] # time-only mask

    # Masking & assignment 
    mask = df['ts'] < pd.Timestamp('2020-01-01')
    df.loc[mask, 'status'] = 'expired' # safe assignment
    sub = df[df['ts'] > pd.Timestamp('2021-01-01')].copy() # copy before mutating slice

#### Lesson 3A: Using Python to Implement the Relational Model

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

#### Lesson 3B: Using Pyhton to Implement the Relational Model (set and reset index)

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

#### Lesson 3C: Using Pyhton to Implement the Relational Model (fast lookups, slicing, group by)

    #!                             name  salary
    #! employee_id department
    #! 101         HR             Alice   60000
    #! 102         Engineering      Bob   80000
    #! 101         Engineering  Charlie   75000
    #! 103         Sales          David   70000

    # 1: Fast lookups. Grab the row for employee 101 in Engineering
    df.loc[(101, 'Engineering')]
    #! name      Charlie
    #! salary      75000
    #! Name: (101, Engineering), dtype: object

    # 2: Slicing on partial keys. All rows for employee 101 across departments
    df.loc[101]
    #!                 name  salary
    #! department
    #! HR             Alice   60000
    #! Engineering  Charlie   75000

    # 3: Groupby on index levels. Average salary per department
    df.groupby(level=1).mean(numeric_only=True)
    #!                   salary
    #! department
    #! Engineering  77500.0
    #! HR           60000.0
    #! Sales        70000.0

#### Lesson 3D: Using Pyhton to Implement the Relational Model (union join aka full outer join)

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

#### Lesson 3E: Using Pyhton to Implement the Relational Model (left join)

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

#### Lesson 3F: Using Pyhton to Implement the Relational Model (inner join)

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

#### Lesson 4: Group 

    df = df.set_index(['plan_id', 'mobile'])
    df = df.groupby(level=['plan_id', 'mobile']).agg(
        nmbr_mac=('mac', 'count'),
        # other aggs, if any
    )
    # Inspect what we've aggregated
    df['nmbr_mac'].value_counts().sort_index()

    # NOTE: 
    # - Available aggs: count, nunique, min, max, first, last, sum, mean,
    #   median, mode

#### Lesson 5: Feature Engineering (Creating Helper Columns)

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

#### Lesson 6A: Pivot Table Definition

    df
    #!   region prod   Q1     Q2     Q3
    #! 0      N    A  100  110.0  120.0
    #! 1      N    B  150  160.0    NaN
    #! 2      S    A  200  210.0  220.0
    #! 3      S    B  250  260.0  270.0
    #! 4      E    A  300    NaN  320.0
    #! 5      E    B  350  360.0  370.0

    # Given a relation (table) T with attributes {R_attrs} (row keys), 
    # {C_attrs} (column keys), and {V} (value(s)), and an aggregation function 
    # agg, the pivot table P is the function:
    #   P(r, c) = agg({ t.V | t in T and t.R_attrs = r and t.C_attrs = c }),
    #   where r ranges over unique values of R_attrs and c over unique values
    #   of C_attrs

    pivot = pd.pivot_table(
        df, # T
        index='region', # {R_attrs}; 
        columns='prod', # {C_attrs}
        values=['Q1', 'Q2', 'Q3'], # {V}
        aggfunc='mean', # agg
        fill_value=0, # considers NaN values to be 0
    )
    # NOTE: Also - whether or not you set_index from before doesn't affect the
    # .pivot_table method. You need to specify the index param nevertheless.
    pivot = pivot.sort_index(axis=1)
    print(pivot)

    #!             Q1            Q2            Q3
    #! prod         A      B      A      B      A      B
    #! region
    #! E        300.0  350.0    0.0  360.0  320.0  370.0
    #! N        100.0  150.0  110.0  160.0  120.0    0.0
    #! S        200.0  250.0  210.0  260.0  220.0  270.0

#### Lesson 6B: Pivot Table (Formatting)

    # Now, because the Relational Model mandates that all relational functions
    # output a relation, lets format the pivot table to be a useful relational df
    # consistent with the indexing of its input for subsequent processing

    pivot.columns = [f"{val}_{col}" for val, col in pivot.columns]
    print(pivot)

    #!          Q1_A   Q1_B   Q2_A   Q2_B   Q3_A   Q3_B
    #! region
    #! E       300.0  350.0    0.0  360.0  320.0  370.0
    #! N       100.0  150.0  110.0  160.0  120.0    0.0
    #! S       200.0  250.0  210.0  260.0  220.0  270.0

#### Lesson 6C: Pivot Table (Motivation x Ability Grid)

    # Goal: 3x3 table with cols: motivation, high_ability, med_ability, low_ability

    # Assume df has:
    # - util_rng_qc: 1-10 (utilisation quantile, 10 = highest usage)
    # - churn_risk_qc: 1-10 (churn risk quantile, 10 = highest risk)

    df['motivation'] = np.where(df['churn_risk_qc'] <= 3, 'high', np.where(df['churn_risk_qc'] <= 7, 'med', 'low'))
    df['ability'] = np.where(df['util_rng_qc'] >= 9, 'high_ability', np.where(df['util_rng_qc'] >= 4, 'med_ability', 'low_ability'))

    df = ( 
        df.groupby(['motivation', 'ability'])
          .agg(users=('plan_id', 'nunique'))
          .reset_index() 
    )
    # NOTE: 
    # - We wrap it in a (), to allow us to indent each .method on a seperate line
    # - We need to reset index here, so that the pivot operation that follows can access the ability column values 

    # Pivot so abilities become columns and motivations become rows; fill 
    # missing with 0 and force desired order
    pk_motivation_df = ( 
        pk_motivation_ability_df
          .pivot(index='motivation', columns='ability', values='users')
          .fillna(0) # missing combos -> 0 users
          .reindex(['high', 'med', 'low']) # enforce row order
          .reindex(['high_ability', 'med_ability', 'low_ability'], axis=1) # enforce column order/names
          .reset_index() # turn 'motivation' back into a column
    )

    # Flatten the columns so 'ability' doesn't sit on top
    pk_motivation_df.columns.name = None  
    print(pk_motivation_df)
