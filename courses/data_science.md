# Data Science

## Part I: Pandas

### Section 1: Crash Course

#### Lesson 1: Inspecting Dataframes

	# Get columns as list
	df.columns.tolist()

	# Get tuple showing (row_count, col_count)
	df.shape

	# Preview first 5 rows
	df.head()

	# Get count of distinct non-null values
	df['mac'].nunique()
	df[['mac', 'plan_id', 'mobile']].nunique() # per-column counts excluding NaNs
	df['plan_duration'].value_counts() # Returns Pandas series instead

	# Lowercase all column names
	df.columns = df.columns.str.lower()

	# Create composite key string
	df['key'] = df['mac'].astype(str) + '_' + df['mobile'].astype(str) + '_' + df['plan_id'].astype(str)

#### Lesson 2: Handling Datetimes and Calculations

    df['plan_start_time'] = pd.to_datetime(df['plan_start_time'], errors='coerce')
    df['plan_end_time'] = pd.to_datetime(df['plan_end_time'], errors='coerce')
    # Pandas does not store stuff as bulky Python datetime objects
    # - Instead, it parses str time columns as the numpy type datetime64[ns]
    # - coerce makes invalids -> NaT

    # Get diff of days as integer
    df['plan_duration'] = (df['plan_end_time'] - df['plan_start_time']).dt.days
    # NOTE:
    # - (end - start) yields a Timedelta
    # - .dt.days gives the integer day component
    # - NaT propagates to NaN after .dt.days. Thus, if any NaT, all become float

#### Lesson 3: Group By

    # Group and peek
    group = df.groupby(['plan_id', 'mobile']).agg(nmbr_mac=('mac', 'count')).reset_index()
    group.head()

    summary = df.groupby('number_days').agg(
        count_all=('mobile', 'count'),
        median_total_time_spent_all=('total_time_spent', 'median')
    ).reset_index()

    # Additionally append column using the aggregated count_all to compute % of unique values
    total_uniques = df['mobile'].nunique()
    summary['pct_of_unique_mobiles_all'] = (summary['count_all'] / total_uniques) * 100

#### Lesson 4A: Feature Engineering

    # Feature engineering: Discretizing continuous variable columns into bins
	bins = [0, 10, 20, 28, 35, float('inf')]
	df['days_rng'] = pd.cut(df['number_days'], bins=bins, labels=False)
	df['days_rng'].value_counts()
	# NOTE: pd.cut gives fixed, value-based bins, mapping them to indices: 0 -> (0,10], 1 -> (10,20] ... 

	df['utilisation'] = df['number_days'] / df['plan_duration']
	df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0)
	df['mac_80%'] = np.where((df['utilisation'] > 0.8) & (df['utilisation'] <= 0.9), 1, 0)

    df['util_range'] = pd.qcut(df['utilisation'], q=10)
    # NOTE: pd.qcut gives quantile / equal-frequency bins cut by data quantiles so bins have ~equal counts
    # Lets now check the freq of rows across all bins
    df.groupby('util_range',observed=True).size().reset_index()

#### Lesson 4B: Feature Engineering

    # Aggregation based on engineered features
	df.groupby('util_range').agg(freq=('mac', 'count'),
	   						  min_util=('utilisation', 'min'),
	   						  max_util=('utilisation', 'max'),
	   						  mean_util=('utilisation', 'mean')).reset_index()

	df.groupby('days_rng').agg(freq=('mobile', 'count'),
								min_days=('number_days', 'min'),
								max_days=('number_days', 'max'),
								mean_days=('number_days', 'mean')).reset_index()

    # Change 'Primary Key' (plan_id+mobile), then re-bin & re-summarize
	group = df.groupby(['plan_id', 'mobile'])['number_days'].max().reset_index()
	group['days_rng'] = pd.cut(group['number_days'], bins=bins, labels=False) + 1
	group.groupby('days_rng').agg(freq=('mobile', 'count'),
								  min_days=('number_days', 'min'),
								  max_days=('number_days', 'max'),
								  mean_days=('number_days', 'mean')).reset_index()
    # Note: In data science, the 'Primary Key' of a Relation, is essentially its
    # 'Unit of Analysis'

### Section 2: pandasql

#### Lesson 1: Inspecting Dataframes

    # Get columns 
    sqldf("PRAGMA table_info(df);")

    # Get row count 
    sqldf("SELECT COUNT(*) AS row_count FROM df")

    # Preview first 5 rows
    sqldf("SELECT * FROM df LIMIT 5")

    # Get count of distinct non-null values
    sqldf("SELECT COUNT(DISTINCT mac) AS nmbr FROM df")
    sqldf("SELECT COUNT(DISTINCT mac) AS mac, COUNT(DISTINCT plan_id) AS plan_id, COUNT(DISTINCT mobile) AS mobile FROM df")

    # value_counts equivalent (returns dataframe similar to series)
    sqldf("SELECT plan_duration, COUNT(*) AS count FROM df GROUP BY plan_duration ORDER BY count DESC")

    # Create composite key string
    df = sqldf("SELECT *, CAST(mac AS TEXT) || '_' || CAST(mobile AS TEXT) || '_' || CAST(plan_id AS TEXT) AS key FROM df")

#### Lesson 2: Handling Datetimes and Calculations

    # Handling datetimes: SQLite parses strings on the fly with julianday() for calculations.
    # Invalid dates become NULL (NaN in pandas). No explicit to_datetime needed, but assumes parseable format.

    # Get diff of days as integer (adds column)
    df = sqldf("""
        SELECT *, CAST((julianday(plan_end_time) - julianday(plan_start_time)) AS INTEGER) AS plan_duration
        FROM df
    """)
    # NOTE:
    # - julianday() converts to Julian day number (float); difference gives days.
    # - CAST to INTEGER floors to whole days, like .dt.days.
    # - NULL propagates if parse fails.

#### Lesson 3: Group By

    # Group and peek
    group = sqldf("""
        SELECT plan_id, mobile, COUNT(mac) AS nmbr_mac
        FROM df
        GROUP BY plan_id, mobile
    """)
    group.head()

    # Decide to aggregate by mobile count after checking its distinct value count
    total_uniques = sqldf("SELECT COUNT(DISTINCT mobile) AS total FROM df")['total'][0]
    summary = sqldf("""
        SELECT number_days, COUNT(mobile) AS count_all, MEDIAN(total_time_spent) AS median_total_time_spent_all
        FROM df
        GROUP BY number_days
    """)

    # Additionally append column using the aggregated count_all to compute % of unique values (mix with Python)
    summary['pct_of_unique_mobiles_all'] = (summary['count_all'] / total_uniques) * 100

#### Lesson 4A: Feature Engineering

    # Feature engineering: Discretizing continuous variable columns into bins (fixed bins via CASE)
    df = sqldf("""
        SELECT *, CASE
            WHEN number_days <= 0 THEN NULL
            WHEN number_days <= 10 THEN 0
            WHEN number_days <= 20 THEN 1
            WHEN number_days <= 28 THEN 2
            WHEN number_days <= 35 THEN 3
            ELSE 4
        END AS days_rng
        FROM df
    """)
    sqldf("SELECT days_rng, COUNT(*) AS count FROM df GROUP BY days_rng")  # value_counts equivalent
    # NOTE: This mimics pd.cut with fixed bins; NULL for <=0 or NaN.

    df = sqldf("SELECT *, CAST(number_days AS REAL) / plan_duration AS utilisation FROM df")

    df = sqldf("""
        SELECT *, CASE WHEN utilisation > 0.9 THEN 1 ELSE 0 END AS mac_90p
        FROM df
    """)
    df = sqldf("""
        SELECT *, CASE WHEN utilisation > 0.8 AND utilisation <= 0.9 THEN 1 ELSE 0 END AS mac_80p
        FROM df
    """)

#### Lesson 4B: Feature Engineering

    # Quantile bins (mimics pd.qcut using NTILE window function)
    df = sqldf("""
        SELECT *, NTILE(10) OVER (ORDER BY utilisation) - 1 AS util_range
        FROM df
        ORDER BY utilisation
    """)
    # NOTE: NTILE(10) divides into 10 roughly equal groups (1-10), -1 makes 0-9.
    # Handles ties by sequential assignment; no exact 'duplicates=drop' but close enough for most data.
    # Assumes SQLite supports windows (it does in modern versions).

#### Lesson 4C: Feature Engineering

    # Aggregation based on engineered features
    sqldf("""
        SELECT util_range, COUNT(mac) AS freq,
               MIN(utilisation) AS min_util,
               MAX(utilisation) AS max_util,
               AVG(utilisation) AS mean_util
        FROM df
        GROUP BY util_range
    """)

    sqldf("""
        SELECT days_rng, COUNT(mobile) AS freq,
               MIN(number_days) AS min_days,
               MAX(number_days) AS max_days,
               AVG(number_days) AS mean_days
        FROM df
        GROUP BY days_rng
    """)

#### Lesson 4D: Feature Engineering

    # Change 'Primary Key' (plan_id+mobile), then re-bin & re-summarize
    group = sqldf("""
        SELECT plan_id, mobile, MAX(number_days) AS number_days,
               CASE
                   WHEN MAX(number_days) <= 0 THEN NULL
                   WHEN MAX(number_days) <= 10 THEN 1
                   WHEN MAX(number_days) <= 20 THEN 2
                   WHEN MAX(number_days) <= 28 THEN 3
                   WHEN MAX(number_days) <= 35 THEN 4
                   ELSE 5
               END AS days_rng
        FROM df
        GROUP BY plan_id, mobile
    """)
    sqldf("""
        SELECT days_rng, COUNT(mobile) AS freq,
               MIN(number_days) AS min_days,
               MAX(number_days) AS max_days,
               AVG(number_days) AS mean_days
        FROM group
        GROUP BY days_rng
    """)
    # Note: In data science, the 'Primary Key' of a Relation, is essentially its
    # 'Unit of Analysis'. Binning adjusted +1 to make 1-based as in pandas example.
