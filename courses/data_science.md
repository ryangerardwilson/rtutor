# Data Science

## Part I: Pandas

### Section 1: Crash Course

#### Lesson 1: Inspecting Dataframes

    df.columns.tolist()
    df.shape
    df['mac'].nunique()
    df[['mac', 'plan_id', 'mobile']].nunique()
    df.columns = df.columns.str.lower()
    df.columns.tolist()

#### Lesson 2: Handling Datetimes and Calculations

    df['plan_start_time'] = pd.to_datetime(df['plan_start_time'], errors='coerce')
    df['plan_end_time'] = pd.to_datetime(df['plan_end_time'], errors='coerce')
    df['plan_duration'] = (df['plan_end_time'] - df['plan_start_time']).dt.days
    df['plan_duration'].value_counts()
    df['key'] = df['mac'].astype(str) + '_' + df['mobile'].astype(str) + '_' + df['plan_id'].astype(str)
    df['key'].nunique()

#### Lesson 3: Cutting into Bins

    from math import inf
    bins = [0, 10, 20, 28, 35, float('inf')]
    df['days_rng'] = pd.cut(df['number_days'], bins=bins, labels=False) + 1
    df['days_rng'].value_counts()

    df['utilisation'] = df['number_days'] / df['plan_duration']
    df['mac_90%'] = np.where(df['utilisation'] > 0.9, 1, 0)
    df['mac_80%'] = np.where((df['utilisation'] > 0.8) & (df['utilisation'] <= 0.9), 1, 0)
    df['util_range'] = pd.qcut(df['utilisation'], q=10, labels=False, duplicates='drop') + 1
    df.groupby('util_range').agg(freq=('mac', 'count'), min_util=('utilisation', 'min'), max_util=('utilisation', 'max'), mean_util=('utilisation', 'mean')).reset_index()

    df.groupby('days_rng').agg(freq=('mobile', 'count'), min_days=('number_days', 'min'), max_days=('number_days', 'max'), mean_days=('number_days', 'mean')).reset_index()
    group = df.groupby(['plan_id', 'mobile'])['number_days'].max().reset_index()
    group['days_rng'] = pd.cut(group['number_days'], bins=bins, labels=False) + 1
    group.groupby('days_rng').agg(freq=('mobile', 'count'), min_days=('number_days', 'min'), max_days=('number_days', 'max'), mean_days=('number_days', 'mean')).reset_index()

#### Lesson 4: Advanced Aggregations

    group = df.groupby(['plan_id', 'mobile']).agg(nmbr_mac=('mac', 'count')).reset_index()
    group.head()

    total_uniques = df['mobile'].nunique()
    summary = df.groupby('number_days').agg(
        count_all=('mobile', 'count'),
        median_total_time_spent_all=('total_time_spent', 'median')
    ).reset_index()
    summary['pct_of_unique_mobiles_all'] = (summary['count_all'] / total_uniques) * 100
