# Data Science

## Part I: Replace Excel with Python

### Section 1: Vanilla Numpy & Pandas

#### Lesson 1: Relational Calculus Thinking

    # A table/dataframe is a way to represent an n-ary mathematical relation, where
    # - n represents the number of columns,
    # - columns represent attributes of tuple indices, and
    # - rows represent a set of {tuples}.
    # Table = { (c1, c2, ..., cn) | each ci in `domain_i`, for i=1 to n }

    # Define the domains implicitly through data types and values
    table = {
        'ID': [1, 2, 3, 3], # Domain: positive integers
        'Name': ['Alice', 'Bob', 'Charlie', 'Charlie'], # Domain: strings
        'Salary': [100000.0, 120000.0, 90000.0, 90000.0] # Domain: non-negative floats
    }

    # While a df, can accomodate duplicate rows - we cannot call such a table a
    # relational table because, rows MUST represent a set of {tuples}. A
    # df representing a relation must have at least 1 key candidate.
    relational_table = table.drop_duplicates()
    pk_id_df = pd.DataFrame(table)
    n = len(pk_id_df.columns)
    columns = pk_id_df.columns

#### Lesson 2: Inspecting Dataframes

	# Get columns
	df.columns
	# Get tuple showing (row_count, col_count)
	df.shape
	# Preview first 5 rows
	df.head()

	# Get count of distinct non-null values
	df['mac'].nunique()
	df[['mac', 'plan_id', 'mobile']].nunique() # per-column counts excluding NaNs

    # Quick way to group by column
	df['plan_duration'].value_counts().sort_index() # Returns Pandas series 

	# Lowercase all column names
	df.columns = df.columns.str.lower()

	# Create composite key string
	df['key'] = df['mac'].astype(str) + '_' + df['mobile'].astype(str) + '_' + df['plan_id'].astype(str)

#### Lesson 3A: Filter & Mask

    # Using the [] operator 
    df[((df['plan_duration'] > 12) & (df['status'].isin([6,12,24]))) | (df['plan_type'] == 'promo')]
    df[~df['plan_id'].isin([4, 5])] # exclude those plan_ids
    df = df[df['plan_duration'] > 12] # Mutates the df
    filtered_df = df[df['plan_duration'] > 12].copy() # Get explicit copy
    df[df['mac'].notna()] # keep rows with mac present

    # Using .dropna (which uses .isna internally) 
    # What pandas treats as NA: np.nan, pandas.NaT, None, pd.NA (and other
    # dtype-specific NA representations). isna() returns True for those.
    df.dropna(subset=['mac', 'mobile']) # drop rows missing mac or mobile

    # Using masks: a mask is a boolean Series whose elements are True where the mask condition is satisfied 
    mask = (df['plan_duration'] > 12) & (df['plan_id'] == 3)
    # select rows where mask is True and only these columns
    df.loc[mask, ['mac', 'mobile', 'plan_id']]
    # set plan_duration to 0 for masked rows 
    df.loc[mask, 'plan_duration'] = 0

    # String filters (always set na=False)
    df[df['mobile'].str.contains('555', na=False)]   
    df[df['mac'].str.startswith('aa', na=False)] 

#### Lesson 3B: Filter & Mask (Datetime Columns)

    # Ensure real datetimes — coerce bad input to NaT, then handle
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')  # Don't guess types; coerce invalid -> NaT
    df = df[df['ts'].notna()] # Drop bad dates explicitly

    # Range comparisons / between — readable and vectorized
    df[df['ts'] >= pd.Timestamp('2020-01-01')] # single-side
    df[df['ts'].between('2020-01-01', '2020-01-31')] # inclusive range (clean)

    # DatetimeIndex slicing — fastest and very readable
    df = df.set_index('ts') # set index once for time ops
    df.loc['2020-01-01':'2020-01-31'] # inclusive index slice
    df.between_time('08:00', '17:00') # time-of-day filter on index

    # Component masks with .dt — year/month/weekday/time
    df[df['ts'].dt.year == 2020] # filter by year
    df[df['ts'].dt.month.isin([1,2,3])] # filter months
    df[df['ts'].dt.weekday < 5] # weekday mask (Mon=0)
    df[df['ts'].dt.time.between(pd.to_datetime('08:00').time(), pd.to_datetime('17:00').time())]  # time-only mask

    # Masking & assignment — use .loc or .copy() to avoid SettingWithCopy
    mask = df['ts'] < pd.Timestamp('2020-01-01')
    df.loc[mask, 'status'] = 'expired' # safe assignment
    sub = df[df['ts'] > pd.Timestamp('2021-01-01')].copy() # copy before mutating slice

#### Lesson 4: Handling Datetimes & Group 

    # Handle datetimes
    df['plan_start_time'] = pd.to_datetime(df['plan_start_time'], errors='coerce')
    df['plan_end_time'] = pd.to_datetime(df['plan_end_time'], errors='coerce')
    # Get a new derived attribute by calculating the diff of days as integer
    df['plan_duration'] = (df['plan_end_time'] - df['plan_start_time']).dt.days

    # Group and peek
    pk_plan_id_mobile_df = df.groupby(['plan_id', 'mobile']).agg(
        nmbr_mac=('mac', 'count'),
        # other aggs, if any
    ).reset_index()
    # Inspect what we've aggregated
    pk_plan_id_mobile_df['nmbr_mac'].value_counts().sort_index()
    # NOTE: Available aggs: count, nunique, min, max, first, last, sum, mean,
    # median, mode

    pk_number_days_df = df.groupby('number_days').agg(
        count_all=('mobile', 'count'),
        median_total_time_spent_all=('total_time_spent', 'median')
    ).reset_index()

    # Append additional derived continuous variable column
    total_uniques = df['mobile'].nunique()
    pk_number_days_df['pct_of_unique_mobiles'] = (pk_number_days_df['count_all'] / total_uniques) * 100

    # NOTE: It is good practice to use pk_{primary_keys}_df as indicators for 
    # the primary keys of the df

#### Lesson 5: Feature Engineering (Creating Helper Columns)

    # Feature engineering: Discretizing continuous variable columns into bins
	df['days_rng_bc'] = pd.cut(df['number_days'], bins=[0, 10, 20, 28, 35, float('inf')], labels=False) 
    # Inspect what we've labelled
	df['days_rng_bc'].value_counts().sort_index()
    # Filter out junk, based on inspection
    df[df['days_rng_bc'].astype(str) != "(35.0, inf]"]
    # NOTE: labels=False gives us the index number of the label (which can
    # directly be used as a numeric feature), instead of the label itself. Don't
    # add this param if you want the col to be more human readable, instead.

    # Append additional derived continuous variable columns
	df['utilisation'] = df['number_days'] / df['plan_duration']

    # Append boolean attribute columns
	df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0) # The first param is basically a Boolean Series mask
	df['mac_80%'] = np.where((df['utilisation'] > 0.8) & (df['utilisation'] <= 0.9), 1, 0)

    # Append quantile bin classification column
    df['util_range_qbc'] = pd.qcut(df['utilisation'],q=10,duplicates='drop',labels=False)
    # Inspect what we've labelled
	df['util_range_qbc'].value_counts().sort_index()
    # NOTE: pd.qcut gives quantile / equal-frequency bins cut by data 
    # quantiles so bins have ~equal counts. Here, we drop duplicates to
    # merge duplicate bins caused by too many duplicate values  

    # For detailed analysis use .groupby to analyze by pivoting the pk: pk_util_range_qbc_df, pk_days_rng_bc_df

    # NOTE: It is good practice to use _bc and _qbc as indicators for 'bin
    # classification' and 'quantile bin classification', respectively

#### Lesson 6A: Pivot Table Definition

    df = pd.DataFrame({
            'region': ['N', 'N', 'S', 'S', 'E', 'E'],
            'prod': ['A', 'B', 'A', 'B', 'A', 'B'],
            'Q1': [100, 150, 200, 250, 300, 350],
            'Q2': [110, 160, 210, 260, np.nan, 360],
            'Q3': [120, np.nan, 220, 270, 320, 370],
        })

    # Definition of a Pivot Table in Relational Calculus
    # Given a relation (table) T with attributes {R_attrs} (row keys), 
    # {C_attrs} (column keys), and {V} (value(s)), and an aggregation function 
    # agg, the pivot table P is the function:
    #   P(r, c) = agg({ t.V | t in T and t.R_attrs = r and t.C_attrs = c }),
    #   where r ranges over unique values of R_attrs and c over unique values
    #   of C_attrs.

    pivot = pd.pivot_table(
        df, # T
        index='region', # {R_attrs}
        columns='prod', # {C_attrs}
        values=['Q1', 'Q2', 'Q3'], # {V}
        aggfunc='mean', # agg
        fill_value=0, # considers NaN values to be 0
    )
    pivot = pivot.sort_index(axis=1)
    print(pivot)

    #             Q1            Q2            Q3
    # prod         A      B      A      B      A      B
    # region
    # E        300.0  350.0    0.0  360.0  320.0  370.0
    # N        100.0  150.0  110.0  160.0  120.0    0.0
    # S        200.0  250.0  210.0  260.0  220.0  270.0

#### Lesson 6B: Pivot Table (Formatting)

    # Now, because the Relation Model mandates that all relational functions
    # output a relation, lets format the pivot table to be a useful relational df
    # consistent with the indexing of its input for subsequent processing

    pk_region_df = pivot.copy()
    pk_region_df.columns = [f"{val}_{col}" for val, col in pk_region_df.columns]
    pk_region_df = pivot.reset_index()
    print(pk_region_df)

    #   region  Q1_A   Q1_B   Q2_A   Q2_B   Q3_A   Q3_B
    # 0      E  300.0  350.0    0.0  360.0  320.0  370.0
    # 1      N  100.0  150.0  110.0  160.0  120.0    0.0
    # 2      S  200.0  250.0  210.0  260.0  220.0  270.0

#### Lesson 5C: Pivot Table (Motivation x Ability Grid)

    # Goal: 3x3 table -> motivation (col), high_ability, med_ability, low_ability

    # Assume df has:
    #   - util_rng_qc     : 1-10 (utilisation quantile, 10 = highest usage)
    #   - churn_risk_qc   : 1-10 (churn risk quantile, 10 = highest risk)

    df['motivation'] = np.where(
        df['churn_risk_qc'] <= 3, 'high', 
        np.where(df['churn_risk_qc'] <= 7, 'med', 'low')
    )

    df['ability'] = np.where( 
        df['util_rng_qc'] >= 9, 'high_ability',
        np.where(df['util_rng_qc'] >= 4, 'med_ability', 'low_ability'),
    )

    pk_motivation_ability_df = ( 
        df.groupby(['motivation', 'ability'])
          .agg(users=('plan_id', 'nunique'))
          .reset_index()  
    )
    # NOTE: We wrap it in a (), to allow us to indent each .method on a separate line

    # Pivot so abilities become columns and motivations become rows; fill missing with 0 and force desired order
    pk_motivation_df = ( 
        pk_motivation_ability_df
          .pivot(index='motivation', columns='ability', values='users')
          .fillna(0)  # missing combos -> 0 users
          .reindex(['high', 'med', 'low'])  # enforce row order
          .reindex(['high_ability', 'med_ability', 'low_ability'], axis=1)  # enforce column order/names
          .reset_index()  # turn 'motivation' back into a column
    )

    # Flatten the columns so "ability" doesn't sit on top
    pk_motivation_df.columns.name = None  
    print(pk_motivation_df)
