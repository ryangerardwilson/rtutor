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

    # Decide to aggregate by mobile count after checking its distinct value count
    total_uniques = df['mobile'].nunique()
    summary = df.groupby('number_days').agg(
        count_all=('mobile', 'count'),
        median_total_time_spent_all=('total_time_spent', 'median')
    ).reset_index()

    # Additionally append column using the aggregated count_all to compute % of unique values
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

    df['util_range'] = pd.qcut(df['utilisation'], q=10, labels=False, duplicates='drop')
    # NOTE: pd.qcut gives quantile / equal-frequency bins cut by data quantiles so bins 
    # have ~equal counts
    # - q -> number of bins. Bin edges are data-derived and therefore unevenly spaced.
    # - labels=False -> integer bin codes 0..n_bins-1 (0 = lowest decile)
    # - duplicates='drop' does NOT remove duplicate rows or values. It tells qcut to drop 
    #   duplicate bin edges that arise when quantile cut points are identical (this happens 
    #   when many values are tied). That reduces the number of bins below q
    # - right=True makes the intervals right-closed (a, b] 

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
