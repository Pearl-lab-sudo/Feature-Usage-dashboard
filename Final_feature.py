import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import psycopg2
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Ladder brand colors
LADDER_COLORS = {
    'primary_blue': '#1E3A8A',
    'secondary_blue': '#3B82F6',
    'orange': '#F97316',
    'yellow': '#FCD34D',
    'white': '#FFFFFF',
    'light_gray': '#F8FAFC',
    'dark_gray': '#64748B'
}

# Page configuration
st.set_page_config(
    page_title="Ladder Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(90deg, {LADDER_COLORS['primary_blue']} 0%, {LADDER_COLORS['secondary_blue']} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }}
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid {LADDER_COLORS['orange']};
        margin-bottom: 1rem;
    }}
    .feature-card {{
        background: {LADDER_COLORS['light_gray']};
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid {LADDER_COLORS['secondary_blue']};
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🚀 Ladder Marketing Analytics Dashboard</h1>
    <p>Comprehensive insights into user engagement and feature adoption</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("### 📅 Date Range Filters")
col1, col2 = st.sidebar.columns(2)
with col1:
    signup_start = st.date_input("Signup Period Start", value=datetime(2025, 7, 17))
with col2:
    signup_end = st.date_input("Signup Period End", value=datetime.now().date())

st.sidebar.markdown("### 🎯 Feature Activity Period")
col3, col4 = st.sidebar.columns(2)
with col3:
    activity_start = st.date_input("Activity Start", value=datetime(2025, 8, 9))
with col4:
    activity_end = st.date_input("Activity End", value=datetime(2025, 8, 13))

# DB connection
def get_database_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# Fetch per-user feature flags
@st.cache_data(ttl=300)
def fetch_user_feature_flags(signup_start, signup_end, activity_start, activity_end):
    conn = get_database_connection()
    if conn is None:
        return pd.DataFrame()

    query = """
    WITH recent_signups AS (
        SELECT id::TEXT AS user_id
        FROM users
        WHERE DATE(created_at) BETWEEN %s AND %s
          AND restricted = false
    ),
    spending_users AS (
        SELECT DISTINCT user_id::TEXT AS user_id
        FROM budgets
        WHERE updated_at BETWEEN %s AND %s
        UNION
        SELECT DISTINCT user_id::TEXT AS user_id
        FROM manual_and_external_transactions
        WHERE created_at BETWEEN %s AND %s
    ),
    investment_users AS (
        SELECT DISTINCT ip.user_id::TEXT AS user_id
        FROM transactions t
        JOIN investment_plans ip ON ip.id = t.investment_plan_id
        WHERE t.status = 'success'
          AND t.updated_at BETWEEN %s AND %s
    ),
    savings_users AS (
        SELECT DISTINCT p.user_id::TEXT AS user_id
        FROM transactions t
        JOIN plans p ON p.id = t.plan_id
        WHERE t.status = 'success'
          AND t.updated_at BETWEEN %s AND %s
    ),
    lady_ai_users AS (
        SELECT DISTINCT "user"::TEXT AS user_id
        FROM slack_message_dump
        WHERE created_at BETWEEN %s AND %s
    )
    SELECT
        rs.user_id AS user_id,
        CASE WHEN su.user_id IS NOT NULL THEN 1 ELSE 0 END AS spending_flag,
        CASE WHEN iu.user_id IS NOT NULL THEN 1 ELSE 0 END AS investment_flag,
        CASE WHEN sv.user_id IS NOT NULL THEN 1 ELSE 0 END AS savings_flag,
        CASE WHEN la.user_id IS NOT NULL THEN 1 ELSE 0 END AS lady_ai_flag
    FROM recent_signups rs
    LEFT JOIN spending_users su ON rs.user_id = su.user_id
    LEFT JOIN investment_users iu ON rs.user_id = iu.user_id
    LEFT JOIN savings_users sv ON rs.user_id = sv.user_id
    LEFT JOIN lady_ai_users la ON rs.user_id = la.user_id;
    """
    params = [
        signup_start, signup_end,  # recent_signups
        activity_start, activity_end,  # spending budgets
        activity_start, activity_end,  # spending transactions
        activity_start, activity_end,  # investment
        activity_start, activity_end,  # savings
        activity_start, activity_end   # lady AI
    ]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Force column names to lowercase (avoids weird renaming)
    df.columns = [c.lower() for c in df.columns]

    return df

# Get data
df_flags = fetch_user_feature_flags(signup_start, signup_end, activity_start, activity_end)

if 'user_id' not in df_flags.columns:
    st.error(f"Missing user_id column in df_flags. Columns: {list(df_flags.columns)}")
    st.stop()

features = ['spending_flag', 'investment_flag', 'savings_flag', 'lady_ai_flag']
feature_names = ['Spending', 'Investment', 'Savings', 'Lady AI']

matrix_values = []
overlap_users = {}

for i, feat_a in enumerate(features):
    row_vals = []
    users_a = df_flags[df_flags[feat_a] == 1]
    count_a = len(users_a)

    for j, feat_b in enumerate(features):
        if count_a > 0:
            users_both = users_a[users_a[feat_b] == 1]
            overlap_users[(feature_names[i], feature_names[j])] = users_both['user_id'].tolist()
            pct = len(users_both) / count_a * 100
        else:
            overlap_users[(feature_names[i], feature_names[j])] = []
            pct = 0
        row_vals.append(round(pct, 1))

    matrix_values.append(row_vals)


if df_flags.empty:
    st.warning("No data found for selected period.")
    st.stop()

# KPI Calculations
total_signups = len(df_flags)
spending_users = df_flags['spending_flag'].sum()
investment_users = df_flags['investment_flag'].sum()
savings_users = df_flags['savings_flag'].sum()
lady_ai_users = df_flags['lady_ai_flag'].sum()
total_active_users = (df_flags[['spending_flag','investment_flag','savings_flag','lady_ai_flag']].sum(axis=1) > 0).sum()
activation_rate = (total_active_users / total_signups) * 100
avg_features = df_flags[['spending_flag','investment_flag','savings_flag','lady_ai_flag']].sum().sum() / total_active_users

# KPIs
st.markdown("## 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total New Signups", f"{total_signups:,}")
col2.metric("Total Active Users", f"{total_active_users:,}")
col3.metric("Activation Rate", f"{activation_rate:.1f}%")
col4.metric("Avg Features/User", f"{avg_features:.1f}")

# Feature usage breakdown
feature_data = pd.DataFrame({
    'feature': ['Spending', 'Savings', 'Investment', 'Lady AI'],
    'users': [spending_users, savings_users, investment_users, lady_ai_users]
})
fig_features = px.bar(
    feature_data.sort_values('users', ascending=True),
    x='users', y='feature', orientation='h',
    title="Active Users by Feature", text='users'
)
st.plotly_chart(fig_features, use_container_width=True)

# 📌 Correlation Heatmap
st.markdown("## 🔍 Feature Usage Correlation")
corr_matrix = df_flags[['spending_flag','investment_flag','savings_flag','lady_ai_flag']].corr()
fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu')
st.plotly_chart(fig_corr, use_container_width=True)

# 📌 Overlap Analysis
st.markdown("## 🔗 Feature Overlap Analysis")
df_flags['feature_combo'] = df_flags.apply(lambda row: ', '.join(
    [feat for feat, flag in zip(['Spending','Investment','Savings','Lady AI'], row[1:]) if flag == 1]
), axis=1)
overlap_counts = df_flags['feature_combo'].value_counts().reset_index()
overlap_counts.columns = ['Feature Combination', 'Users']
fig_overlap = px.bar(overlap_counts.head(10), x='Users', y='Feature Combination',
                     orientation='h', title="Top Feature Combinations", color='Users')
st.plotly_chart(fig_overlap, use_container_width=True)

# # 📌 Cross-Feature Adoption Rate with Clickable Drill-down
# st.markdown("## 🔄 Cross-Feature Adoption Rate (Interactive)")

# features = ['spending_flag', 'investment_flag', 'savings_flag', 'lady_ai_flag']
# feature_names = ['Spending', 'Investment', 'Savings', 'Lady AI']

# # Prepare matrix values + user overlap dictionary
# matrix_values = []
# overlap_users = {}

# for i, feat_a in enumerate(features):
#     row_vals = []
#     users_a = df_flags[df_flags[feat_a] == 1]
#     count_a = len(users_a)
#     for j, feat_b in enumerate(features):
#         if count_a > 0:
#             users_both = users_a[users_a[feat_b] == 1]
#             pct = len(users_both) / count_a * 100
#         else:
#             pct = 0
#             users_both = pd.DataFrame()
#         row_vals.append(round(pct, 1))
#         overlap_users[(feature_names[i], feature_names[j])] = users_both['user_id'].tolist()
#     matrix_values.append(row_vals)

# # Build heatmap
# fig = go.Figure(data=go.Heatmap(
#     z=matrix_values,
#     x=feature_names,
#     y=feature_names,
#     colorscale='Blues',
#     hovertemplate='%{y} → %{x}: %{z:.1f}%<extra></extra>',
#     text=[[f"{pct}%" for pct in row] for row in matrix_values],
#     texttemplate="%{text}"
# ))

# fig.update_layout(
#     title="Cross-Feature Adoption (%)",
#     xaxis_title="Target Feature",
#     yaxis_title="Starting Feature",
#     height=500
# )

# # Render clickable heatmap
# selected = st.plotly_chart(fig, use_container_width=True, on_select="click")

# # Drill-down after click
# if selected and selected["points"]:
#     point = selected["points"][0]
#     row_feature = feature_names[point["y"]]  # starting feature
#     col_feature = feature_names[point["x"]]  # target feature
    
#     users_list = overlap_users.get((row_feature, col_feature), [])
#     st.markdown(f"### 👥 Users who used both `{row_feature}` and `{col_feature}`")
#     st.write(f"**{len(users_list)} users found**")

#     if users_list:
#         st.dataframe(pd.DataFrame({'user_id': users_list}))
#     else:
#         st.info("No matching users found.")
