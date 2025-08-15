import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import psycopg2

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DB_NAME")

# ----------------------------
# Database connection helper
# ----------------------------
def get_database_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# ----------------------------
# Debug function to check signup count
# ----------------------------
@st.cache_data(ttl=300)
def debug_signup_count(signup_start_dt, signup_end_dt):
    conn = get_database_connection()
    if conn is None:
        return 0
    
    query = """
    SELECT COUNT(DISTINCT id) as signup_count
    FROM users
    WHERE DATE(created_at) >= %s AND DATE(created_at) <= %s
      AND restricted = false
    """
    
    try:
        result = pd.read_sql_query(query, conn, params=[signup_start_dt.date(), signup_end_dt.date()])
        return result.iloc[0]['signup_count']
    except Exception as e:
        st.error(f"Debug query error: {e}")
        return 0
    finally:
        conn.close()

# ----------------------------
# Feature Usage Query - FIXED
# ----------------------------
@st.cache_data(ttl=300)
def fetch_user_feature_flags(signup_start_dt, signup_end_dt, activity_start_dt, activity_end_dt):
    conn = get_database_connection()
    if conn is None:
        return pd.DataFrame()

    query = """
    WITH recent_signups AS (
        SELECT id::TEXT AS user_id
        FROM users
        WHERE DATE(created_at) >= %s AND DATE(created_at) <= %s
          AND restricted = false
    )
    SELECT
        rs.user_id,
        -- Spending flag: check if user has any spending activity in the period
        CASE WHEN EXISTS (
            SELECT 1 FROM budgets b 
            WHERE b.user_id::TEXT = rs.user_id 
            AND b.created_at >= %s AND b.created_at < %s
        ) OR EXISTS (
            SELECT 1 FROM manual_and_external_transactions met 
            WHERE met.user_id::TEXT = rs.user_id 
            AND met.created_at >= %s AND met.created_at < %s
        ) THEN 1 ELSE 0 END AS spending_flag,
        
        -- Investment flag
        CASE WHEN EXISTS (
            SELECT 1 FROM transactions t
            JOIN investment_plans ip ON ip.id = t.investment_plan_id
            WHERE ip.user_id::TEXT = rs.user_id
            AND t.status = 'success'
            AND t.updated_at >= %s AND t.updated_at < %s
        ) THEN 1 ELSE 0 END AS investment_flag,
        
        -- Savings flag  
        CASE WHEN EXISTS (
            SELECT 1 FROM transactions t
            JOIN plans p ON p.id = t.plan_id
            WHERE p.user_id::TEXT = rs.user_id
            AND t.status = 'success'
            AND t.updated_at >= %s AND t.updated_at < %s
        ) THEN 1 ELSE 0 END AS savings_flag,
        
        -- Lady AI flag
        CASE WHEN EXISTS (
            SELECT 1 FROM slack_message_dump smd
            WHERE smd."user"::TEXT = rs.user_id
            AND smd.created_at >= %s AND smd.created_at < %s
        ) THEN 1 ELSE 0 END AS lady_ai_flag,
        
        -- Recurring spending flag: more than 1 distinct day of activity
        CASE WHEN (
            SELECT COUNT(DISTINCT DATE(b.created_at))
            FROM budgets b 
            WHERE b.user_id::TEXT = rs.user_id 
            AND b.created_at >= %s AND b.created_at < %s
        ) + (
            SELECT COUNT(DISTINCT DATE(met.created_at))
            FROM manual_and_external_transactions met 
            WHERE met.user_id::TEXT = rs.user_id 
            AND met.created_at >= %s AND met.created_at < %s
        ) > 1 THEN 1 ELSE 0 END AS recurring_spending_flag,
        
        -- Recurring investment flag
        CASE WHEN (
            SELECT COUNT(DISTINCT DATE(t.updated_at))
            FROM transactions t
            JOIN investment_plans ip ON ip.id = t.investment_plan_id
            WHERE ip.user_id::TEXT = rs.user_id
            AND t.status = 'success'
            AND t.updated_at >= %s AND t.updated_at < %s
        ) > 1 THEN 1 ELSE 0 END AS recurring_investment_flag,
        
        -- Recurring savings flag
        CASE WHEN (
            SELECT COUNT(DISTINCT DATE(t.updated_at))
            FROM transactions t
            JOIN plans p ON p.id = t.plan_id
            WHERE p.user_id::TEXT = rs.user_id
            AND t.status = 'success'
            AND t.updated_at >= %s AND t.updated_at < %s
        ) > 1 THEN 1 ELSE 0 END AS recurring_savings_flag,
        
        -- Recurring Lady AI flag
        CASE WHEN (
            SELECT COUNT(DISTINCT DATE(smd.created_at))
            FROM slack_message_dump smd
            WHERE smd."user"::TEXT = rs.user_id
            AND smd.created_at >= %s AND smd.created_at < %s
        ) > 1 THEN 1 ELSE 0 END AS recurring_lady_ai_flag,
        
        -- Recurring any flag: if user has recurring activity in any feature
        CASE WHEN (
            -- Recurring spending
            (SELECT COUNT(DISTINCT DATE(b.created_at))
             FROM budgets b 
             WHERE b.user_id::TEXT = rs.user_id 
             AND b.created_at >= %s AND b.created_at < %s) + 
            (SELECT COUNT(DISTINCT DATE(met.created_at))
             FROM manual_and_external_transactions met 
             WHERE met.user_id::TEXT = rs.user_id 
             AND met.created_at >= %s AND met.created_at < %s) > 1
        ) OR (
            -- Recurring investment
            (SELECT COUNT(DISTINCT DATE(t.updated_at))
             FROM transactions t
             JOIN investment_plans ip ON ip.id = t.investment_plan_id
             WHERE ip.user_id::TEXT = rs.user_id
             AND t.status = 'success'
             AND t.updated_at >= %s AND t.updated_at < %s) > 1
        ) OR (
            -- Recurring savings
            (SELECT COUNT(DISTINCT DATE(t.updated_at))
             FROM transactions t
             JOIN plans p ON p.id = t.plan_id
             WHERE p.user_id::TEXT = rs.user_id
             AND t.status = 'success'
             AND t.updated_at >= %s AND t.updated_at < %s) > 1
        ) OR (
            -- Recurring Lady AI
            (SELECT COUNT(DISTINCT DATE(smd.created_at))
             FROM slack_message_dump smd
             WHERE smd."user"::TEXT = rs.user_id
             AND smd.created_at >= %s AND smd.created_at < %s) > 1
        ) THEN 1 ELSE 0 END AS recurring_any_flag
        
    FROM recent_signups rs
    ORDER BY rs.user_id;
    """
    params = [
        signup_start_dt.date(), signup_end_dt.date(),  # Signup period (2 params)
        # Activity periods - need to repeat 12 times for all the EXISTS and COUNT queries
        activity_start_dt, activity_end_dt,  # spending EXISTS 1
        activity_start_dt, activity_end_dt,  # spending EXISTS 2  
        activity_start_dt, activity_end_dt,  # investment EXISTS
        activity_start_dt, activity_end_dt,  # savings EXISTS
        activity_start_dt, activity_end_dt,  # lady_ai EXISTS
        activity_start_dt, activity_end_dt,  # recurring spending COUNT 1
        activity_start_dt, activity_end_dt,  # recurring spending COUNT 2
        activity_start_dt, activity_end_dt,  # recurring investment COUNT
        activity_start_dt, activity_end_dt,  # recurring savings COUNT
        activity_start_dt, activity_end_dt,  # recurring lady_ai COUNT
        activity_start_dt, activity_end_dt,  # recurring any - spending 1
        activity_start_dt, activity_end_dt,  # recurring any - spending 2
        activity_start_dt, activity_end_dt,  # recurring any - investment
        activity_start_dt, activity_end_dt,  # recurring any - savings
        activity_start_dt, activity_end_dt,  # recurring any - lady_ai
    ]
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ----------------------------
# FFP Analysis Query
# ----------------------------
@st.cache_data(ttl=300)
def load_ffp_data(start_dt, end_dt):
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        ffp_df = pd.read_sql("SELECT * FROM financial_simulator_v2 WHERE created_at >= %s AND created_at < %s", engine, params=(start_dt, end_dt))
        feedback_df = pd.read_sql("SELECT * FROM financial_simulator_reviews WHERE created_at >= %s AND created_at < %s", engine, params=(start_dt, end_dt))
        return ffp_df, feedback_df
    except Exception as e:
        st.error(f"Error loading FFP data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if 'engine' in locals():
            engine.dispose()

def parse_metadata(metadata_str):
    try:
        parsed = json.loads(metadata_str)
        if isinstance(parsed, dict) and "plan" in parsed:
            return {item['question']: item['answer'] for item in parsed['plan'] if isinstance(item, dict)}
    except:
        return {}
    return {}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="User Engagement Dashboard", layout="wide")
st.title("📊 User Engagement Dashboard")

page = st.sidebar.selectbox("Select a Page:", ["📈 Feature Usage Breakdown", "📊 FFP Analysis"])

# ----------------------------
# Page 1: Feature Usage Breakdown
# ----------------------------
if page == "📈 Feature Usage Breakdown":
    st.sidebar.markdown("### 📅 Signup Period")
    col1, col2 = st.sidebar.columns(2)
    signup_start = col1.date_input("Start", value=datetime(2025, 7, 17), key="signup_start")
    signup_end = col2.date_input("End", value=datetime.now().date(), key="signup_end")
    signup_start_dt = datetime.combine(signup_start, datetime.min.time())
    signup_end_dt = datetime.combine(signup_end, datetime.min.time())  # Remove +1 day

    st.sidebar.markdown("### 🎯 Feature Activity Period")
    col3, col4 = st.sidebar.columns(2)
    activity_start = col3.date_input("Start", value=datetime(2025, 8, 9), key="activity_start")
    activity_end = col4.date_input("End", value=datetime(2025, 8, 13), key="activity_end")
    activity_start_dt = datetime.combine(activity_start, datetime.min.time())
    activity_end_dt = datetime.combine(activity_end + timedelta(days=1), datetime.min.time())

    # Debug information
    st.sidebar.markdown("#### Debug Info:")
    st.sidebar.text(f"Signup: {signup_start} to {signup_end}")
    st.sidebar.text(f"Activity: {activity_start} to {activity_end}")
    
    # Add debug signup count
    debug_count = debug_signup_count(signup_start_dt, signup_end_dt)
    st.sidebar.text(f"Expected signups: {debug_count}")

    df_flags = fetch_user_feature_flags(signup_start_dt, signup_end_dt, activity_start_dt, activity_end_dt)
    if df_flags.empty:
        st.warning("No data found for selected period.")
        st.stop()

    total_signups = len(df_flags)
    st.sidebar.text(f"Actual signups: {total_signups}")
    
    # Show difference if any
    if debug_count != total_signups:
        st.sidebar.error(f"MISMATCH! Expected {debug_count}, got {total_signups}")

    total_signups = len(df_flags)
    total_active_users = (df_flags[['spending_flag','investment_flag','savings_flag','lady_ai_flag']].sum(axis=1) > 0).sum()
    activation_rate = (total_active_users / total_signups * 100) if total_signups else 0
    avg_features = df_flags[['spending_flag','investment_flag','savings_flag','lady_ai_flag']].sum().sum() / total_active_users if total_active_users else 0
    total_recurring_users = df_flags['recurring_any_flag'].sum()

    st.markdown("## 📌 Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total New Signups", f"{total_signups:,}")
    k2.metric("Total Active Users", f"{total_active_users:,}")
    k3.metric("Activation Rate", f"{activation_rate:.1f}%")
    k4.metric("Avg Features/User", f"{avg_features:.1f}")
    k5.metric("Recurring Users", f"{total_recurring_users:,}")

    # Feature usage charts
    feature_data = pd.DataFrame({
        'feature': ['Spending', 'Savings', 'Investment', 'Lady AI'],
        'users': [
            df_flags['spending_flag'].sum(),
            df_flags['savings_flag'].sum(),
            df_flags['investment_flag'].sum(),
            df_flags['lady_ai_flag'].sum()
        ]
    })
    st.plotly_chart(px.bar(feature_data.sort_values('users'), x='users', y='feature', orientation='h', text='users', title="Active Users by Feature"), use_container_width=True)

    recurring_data = pd.DataFrame({
        'feature': ['Spending', 'Savings', 'Investment', 'Lady AI'],
        'users': [
            df_flags['recurring_spending_flag'].sum(),
            df_flags['recurring_savings_flag'].sum(),
            df_flags['recurring_investment_flag'].sum(),
            df_flags['recurring_lady_ai_flag'].sum()
        ]
    })
    st.plotly_chart(px.bar(recurring_data.sort_values('users'), x='users', y='feature', orientation='h', text='users', title="Recurring Users by Feature"), use_container_width=True)

    st.markdown("## 🔍 Feature Usage Correlation")
    correlation_matrix = df_flags[['spending_flag','investment_flag','savings_flag','lady_ai_flag']].corr()
    st.plotly_chart(px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='RdBu', title="Feature Usage Correlation Matrix"), use_container_width=True)

    # Feature combinations
    df_flags['feature_combo'] = df_flags.apply(lambda row: ', '.join([feat for feat, flag in zip(['Spending','Investment','Savings','Lady AI'], row[1:5]) if flag == 1]) or 'No Features Used', axis=1)
    overlap_counts = df_flags['feature_combo'].value_counts().reset_index()
    overlap_counts.columns = ['Feature Combination', 'Users']
    st.plotly_chart(px.bar(overlap_counts.head(10), x='Users', y='Feature Combination', orientation='h', color='Users', title="Top Feature Combinations"), use_container_width=True)

    # Cross-feature adoption heatmap
    features = ['spending_flag', 'investment_flag', 'savings_flag', 'lady_ai_flag']
    feature_names = ['Spending', 'Investment', 'Savings', 'Lady AI']
    matrix_values, overlap_users = [], {}
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

    st.plotly_chart(go.Figure(data=go.Heatmap(z=matrix_values, x=feature_names, y=feature_names, colorscale='Blues', text=[[f"{pct}%" for pct in row] for row in matrix_values], texttemplate="%{text}", hovertemplate='%{y} → %{x}: %{z:.1f}%<extra></extra>')).update_layout(title="Cross-Feature Adoption (%)"), use_container_width=True)

    st.markdown("### 👥 User Overlap Drill-Down")
    col_a, col_b = st.columns(2)
    start_feature = col_a.selectbox("Starting Feature", feature_names, key="start_feature")
    target_feature = col_b.selectbox("Target Feature", feature_names, key="target_feature")
    if start_feature and target_feature:
        users_list = overlap_users.get((start_feature, target_feature), [])
        st.write(f"**{len(users_list)} users** used both `{start_feature}` and `{target_feature}`")
        if users_list:
            st.dataframe(pd.DataFrame({'user_id': users_list}))
        else:
            st.info("No matching users found.")

# ----------------------------
# Page 2: FFP Analysis - FIXED
# ----------------------------
elif page == "📊 FFP Analysis":
    st.sidebar.markdown("### 📅 Date Range")
    # Fix: Use a tuple/list input for date range
    date_range = st.sidebar.date_input(
        "Select Date Range", 
        value=[datetime(2025, 7, 17).date(), datetime.now().date()],
        key="ffp_date_range"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0] if date_range else datetime.now().date()
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    ffp_df, feedback_df = load_ffp_data(start_dt, end_dt)
    if ffp_df.empty and feedback_df.empty:
        st.warning("No FFP data found for selected period.")
        st.stop()

    # Convert datetime columns
    if not ffp_df.empty:
        ffp_df['created_at'] = pd.to_datetime(ffp_df['created_at'])
    if not feedback_df.empty:
        feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at'])

    st.title("📈 Ladder Free Financial Plan (FFP) Engagement Dashboard")
    st.markdown("Gain actionable insights into how users interact with the Free Financial Plan experience.")

    # Metrics
    col1, col2 = st.columns(2)
    if not ffp_df.empty:
        parsed_metadata = ffp_df['metadata'].apply(parse_metadata)
        total_completed = parsed_metadata.apply(lambda x: len([v for v in x.values() if v not in (None, '', [], {})]))
        completed_surveys = (total_completed == total_completed.max()).sum() if len(total_completed) > 0 else 0
        col1.metric("✅ Completed Surveys (All Questions)", completed_surveys)
        col2.metric("🔥 Downloads", completed_surveys)  # Assuming downloads = completed surveys
    else:
        col1.metric("✅ Completed Surveys (All Questions)", 0)
        col2.metric("🔥 Downloads", 0)

    st.subheader("📊 Engagement Over Time and User Feedback")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Daily Submissions")
        if not ffp_df.empty:
            trend_df = ffp_df.groupby(ffp_df['created_at'].dt.date).size().reset_index(name='Submissions')
            trend_df = trend_df.rename(columns={"created_at": "Date"})
            st.line_chart(trend_df.set_index("Date"))
        else:
            st.info("No submission data available")
    
    with c2:
        st.subheader("👬 Reactions")
        if not feedback_df.empty:
            reaction_counts = feedback_df['reaction'].value_counts()
            st.bar_chart(reaction_counts)
        else:
            st.info("No reaction data available")

    # FIXED: Add expander for comments
    if not feedback_df.empty and 'comment' in feedback_df.columns:
        with st.expander("💬 View All Comments"):
            for _, row in feedback_df.iterrows():
                if pd.notna(row.get('comment', None)) and row.get('comment', '').strip():
                    st.markdown(f"- **{row['reaction'].capitalize()}** — {row['comment']} *(on {row['created_at'].date()})*")
    else:
        st.info("No comments available for the selected period.")

    st.markdown("""<hr/><center style='color:gray;'>Built by the Data Team @ Ladder • v1.0</center>""", unsafe_allow_html=True)
