# Data Science

## Part I: Pandas

### Section 1: Crash Course

#### Lesson 1: Inspecting Dataframes

	# Get columns
	df.columns
	# Get tuple showing (row_count, col_count)
	df.shape
	# Preview first 5 rows
	df.head()

	# Get count of distinct non-null values
	df['mac'].nunique()
	df[['mac', 'plan_id', 'mobile']].nunique() # per-column counts excluding NaNs

    # Get count of rows of each distinct value of that column
    # (a quick way to group by column)
	df['plan_duration'].value_counts().sort_index() # Returns Pandas series 

	# Lowercase all column names
	df.columns = df.columns.str.lower()

	# Create composite key string
	df['key'] = df['mac'].astype(str) + '_' + df['mobile'].astype(str) + '_' + df['plan_id'].astype(str)

#### Lesson 2: Handling Datetimes & Group 

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
    NOTE: Available aggs: count, nunique, min, max, first, last, sum, mean,
    median, mode

    pk_number_days_df = df.groupby('number_days').agg(
        count_all=('mobile', 'count'),
        median_total_time_spent_all=('total_time_spent', 'median')
    ).reset_index()

    # Append additional derived continuous variable column
    total_uniques = df['mobile'].nunique()
    pk_number_days_df['pct_of_unique_mobiles'] = (pk_number_days_df['count_all'] / total_uniques) * 100

    # NOTE: It is good practice to use pk_{primary_keys}_df as indicators for 
    # the primary keys of the df

#### Lesson 3: Feature Engineering Workflow

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
	df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0)
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
