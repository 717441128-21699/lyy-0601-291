import os
import sys
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import (
    DataImporter,
    ActivityCleaner,
    PaceAnalyzer,
    HeartRateZones,
    RecoveryReminder,
    GoalTracker,
    WeeklyReport
)
from sample_data_generator import SampleDataGenerator

st.set_page_config(
    page_title='跑步健康管理助手',
    page_icon='🏃',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if 'raw_df' not in st.session_state:
    st.session_state.raw_df = pd.DataFrame()
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = pd.DataFrame()
if 'clean_issues' not in st.session_state:
    st.session_state.clean_issues = pd.DataFrame()
if 'pace_analysis' not in st.session_state:
    st.session_state.pace_analysis = {}
if 'hr_analysis' not in st.session_state:
    st.session_state.hr_analysis = {}
if 'recovery_analysis' not in st.session_state:
    st.session_state.recovery_analysis = {}
if 'goal_analysis' not in st.session_state:
    st.session_state.goal_analysis = {}
if 'weekly_report' not in st.session_state:
    st.session_state.weekly_report = {}
if 'user_config' not in st.session_state:
    st.session_state.user_config = {
        'age': 35,
        'resting_hr': 60,
        'max_hr': 185,
        'monthly_goal_km': 150,
        'weekly_goal_km': 40,
        'yearly_goal_km': 1800
    }


def run_analysis_pipeline():
    cfg = st.session_state.user_config

    if st.session_state.raw_df.empty:
        st.warning('请先导入数据或加载示例数据')
        return

    with st.spinner('正在清洗数据...'):
        cleaner = ActivityCleaner()
        clean_df, issues_df = cleaner.clean_data(st.session_state.raw_df)
        st.session_state.clean_df = clean_df
        st.session_state.clean_issues = issues_df

    if not clean_df.empty:
        with st.spinner('正在分析配速和训练负荷...'):
            pace_analyzer = PaceAnalyzer(
                resting_hr=cfg['resting_hr'],
                max_hr=cfg['max_hr']
            )
            st.session_state.clean_df, st.session_state.pace_analysis = pace_analyzer.analyze(clean_df)

        with st.spinner('正在分析心率区间...'):
            hr_zones = HeartRateZones(
                resting_hr=cfg['resting_hr'],
                max_hr=cfg['max_hr'],
                age=cfg['age']
            )
            st.session_state.clean_df, st.session_state.hr_analysis = hr_zones.analyze(st.session_state.clean_df)

        with st.spinner('正在评估恢复状态...'):
            recovery = RecoveryReminder()
            st.session_state.clean_df, st.session_state.recovery_analysis = recovery.analyze(st.session_state.clean_df)

        with st.spinner('正在追踪训练目标...'):
            goals = GoalTracker(
                monthly_distance_goal_km=cfg['monthly_goal_km'],
                weekly_distance_goal_km=cfg['weekly_goal_km'],
                yearly_distance_goal_km=cfg['yearly_goal_km']
            )
            st.session_state.goal_analysis = goals.analyze(st.session_state.clean_df)

        with st.spinner('正在生成周报...'):
            reporter = WeeklyReport()
            st.session_state.weekly_report = reporter.generate(st.session_state.clean_df)

        st.success('分析完成！')


def format_pace(pace_min_km):
    if not pace_min_km or pace_min_km <= 0:
        return '--:--'
    total_seconds = int(round(pace_min_km * 60))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f'{minutes:02d}:{seconds:02d}'


st.markdown('<p class="main-header">🏃 跑步健康管理助手</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">训练记录整理 · 恢复状态监测 · 目标追踪 · 智能周报</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 📊 数据源')
    use_sample = st.checkbox('使用示例数据', value=True)
    uploaded_files = st.file_uploader(
        '上传运动文件 (支持 GPX/TCX/CSV/FIT/JSON)',
        type=['gpx', 'tcx', 'csv', 'fit', 'json'],
        accept_multiple_files=True
    )

    st.markdown('---')
    st.markdown('## ⚙️ 个人配置')
    cfg = st.session_state.user_config
    cfg['age'] = st.number_input('年龄', min_value=10, max_value=100, value=cfg['age'])
    cfg['resting_hr'] = st.number_input('静息心率 (bpm)', min_value=30, max_value=120, value=cfg['resting_hr'])
    cfg['max_hr'] = st.number_input('最大心率 (bpm)', min_value=100, max_value=230, value=cfg['max_hr'])

    st.markdown('### 🎯 训练目标')
    cfg['weekly_goal_km'] = st.number_input('周跑量目标 (km)', min_value=0.0, value=float(cfg['weekly_goal_km']), step=5.0)
    cfg['monthly_goal_km'] = st.number_input('月跑量目标 (km)', min_value=0.0, value=float(cfg['monthly_goal_km']), step=10.0)
    cfg['yearly_goal_km'] = st.number_input('年跑量目标 (km)', min_value=0.0, value=float(cfg['yearly_goal_km']), step=100.0)

    st.markdown('---')
    analyze_btn = st.button('🔄 运行分析', type='primary', use_container_width=True)

    if analyze_btn:
        if use_sample:
            gen = SampleDataGenerator(seed=42)
            st.session_state.raw_df = gen.generate(days=75)
            run_analysis_pipeline()
        elif uploaded_files:
            importer = DataImporter()
            temp_files = []
            for f in uploaded_files:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getvalue())
                    temp_files.append(tmp.name)

            dfs = []
            for tf in temp_files:
                if tf.lower().endswith('.csv'):
                    batch_df = importer.import_csv_batch(tf)
                    if not batch_df.empty:
                        dfs.append(batch_df)
                else:
                    single_df = importer.import_files([tf])
                    if not single_df.empty:
                        dfs.append(single_df)

            if dfs:
                st.session_state.raw_df = pd.concat(dfs, ignore_index=True)
                run_analysis_pipeline()
            else:
                st.error('未能从上传文件中解析出数据')

            for tf in temp_files:
                try:
                    os.unlink(tf)
                except:
                    pass
        else:
            st.warning('请选择示例数据或上传文件')

if use_sample and st.session_state.raw_df.empty:
    gen = SampleDataGenerator(seed=42)
    st.session_state.raw_df = gen.generate(days=75)
    run_analysis_pipeline()

if not st.session_state.clean_df.empty:
    clean_df = st.session_state.clean_df
    pace = st.session_state.pace_analysis
    hr = st.session_state.hr_analysis
    recovery = st.session_state.recovery_analysis
    goals = st.session_state.goal_analysis
    report = st.session_state.weekly_report

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        '📊 数据总览', '🧹 清洗日志', '⏱️ 配速分析',
        '❤️ 心率区间', '💤 恢复提醒', '🎯 目标追踪', '📝 周报'
    ])

    with tab1:
        st.markdown('<p class="section-title">📊 训练数据总览</p>', unsafe_allow_html=True)

        summary = pace.get('summary', {})
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric('总跑步次数', summary.get('total_runs', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card metric-card-green">', unsafe_allow_html=True)
                st.metric('总距离 (km)', f"{summary.get('total_distance_km', 0):.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card metric-card-orange">', unsafe_allow_html=True)
                st.metric('总时长 (小时)', f"{summary.get('total_duration_h', 0):.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card metric-card-blue">', unsafe_allow_html=True)
                st.metric('平均配速', summary.get('avg_pace', '--:--'))
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('')
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric('最佳配速', summary.get('best_pace', '--:--'))
            with col6:
                st.metric('平均每次跑量 (km)', f"{summary.get('avg_distance_per_run_km', 0):.1f}")
            with col7:
                st.metric('累计爬升 (m)', f"{summary.get('total_elevation_m', 0):.0f}")
            with col8:
                st.metric('训练总负荷', f"{summary.get('total_training_load', 0):.0f}")

        st.markdown('')
        st.markdown('<p class="section-title">📅 最近训练记录</p>', unsafe_allow_html=True)

        display_cols = ['date', 'sport_type', 'distance_km', 'duration_min',
                        'avg_pace_min_km', 'avg_hr', 'elevation_gain_m', 'training_load',
                        'sleep_hours', 'injury', 'notes']
        display_df = clean_df[display_cols].copy()
        display_df.columns = ['日期', '运动类型', '距离(km)', '时长(min)',
                               '配速(min/km)', '平均心率', '爬升(m)', '训练负荷',
                               '睡眠(h)', '伤痛', '备注']

        sport_map = {'running': '🏃 跑步', 'cycling': '🚴 骑行', 'strength': '💪 力量', 'other': '📋 其他'}
        display_df['运动类型'] = display_df['运动类型'].map(sport_map).fillna(display_df['运动类型'])
        display_df['配速(min/km)'] = display_df['配速(min/km)'].apply(lambda x: format_pace(x) if pd.notna(x) else '--')
        display_df['距离(km)'] = display_df['距离(km)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
        display_df['时长(min)'] = display_df['时长(min)'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
        display_df['平均心率'] = display_df['平均心率'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else '-')
        display_df['爬升(m)'] = display_df['爬升(m)'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else '-')
        display_df['训练负荷'] = display_df['训练负荷'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else '-')
        display_df['睡眠(h)'] = display_df['睡眠(h)'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')

        st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
        st.caption(f'共 {len(clean_df)} 条记录，显示最近 20 条')

    with tab2:
        st.markdown('<p class="section-title">🧹 数据清洗日志</p>', unsafe_allow_html=True)

        issues = st.session_state.clean_issues
        st.metric('原始记录数', len(st.session_state.raw_df))
        st.metric('清洗后记录数', len(clean_df))
        st.metric('发现问题数', len(issues))

        if not issues.empty:
            st.dataframe(issues, use_container_width=True, hide_index=True)
        else:
            st.info('✅ 数据质量良好，未发现需要修正的问题')

    with tab3:
        st.markdown('<p class="section-title">⏱️ 配速与负荷分析</p>', unsafe_allow_html=True)

        pace_dist = pace.get('pace_distribution', {})
        elev_stats = pace.get('elevation_stats', {})
        hr_stats = pace.get('heart_rate_stats', {})

        col1, col2 = st.columns(2)
        with col1:
            if pace_dist.get('distribution'):
                dist_df = pd.DataFrame([
                    {'配速区间': k, '次数': v}
                    for k, v in pace_dist['distribution'].items() if v > 0
                ])
                if not dist_df.empty:
                    import plotly.express as px
                    fig = px.bar(dist_df, x='配速区间', y='次数',
                                 title='配速分布', color='次数',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown('##### 📊 配速统计')
            st.write(f"中位数配速: **{pace_dist.get('median_pace', '--')}**")
            st.write(f"Q25: {pace_dist.get('pace_q25', '--')} / Q75: {pace_dist.get('pace_q75', '--')}")

        with col2:
            if elev_stats.get('terrain_distribution'):
                terrain_df = pd.DataFrame([
                    {'地形类型': k, '次数': v}
                    for k, v in elev_stats['terrain_distribution'].items() if v > 0
                ])
                if not terrain_df.empty:
                    fig = px.pie(terrain_df, names='地形类型', values='次数',
                                 title='训练地形分布', hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown('##### 🏔️ 爬升统计')
            st.write(f"平均爬升: **{elev_stats.get('avg_elevation_m', 0):.0f} m**")
            st.write(f"最大爬升: **{elev_stats.get('max_elevation_m', 0):.0f} m**")

        st.markdown('')
        st.markdown('##### ❤️ 心率概况')
        if hr_stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('平均心率', f"{hr_stats.get('avg_hr_overall', 0):.0f}" if hr_stats.get('avg_hr_overall') else '-')
            c2.metric('最低平均心率', f"{hr_stats.get('avg_hr_min', 0):.0f}" if hr_stats.get('avg_hr_min') else '-')
            c3.metric('最高平均心率', f"{hr_stats.get('avg_hr_max', 0):.0f}" if hr_stats.get('avg_hr_max') else '-')
            c4.metric('最大心率记录', f"{hr_stats.get('max_hr_overall', 0):.0f}" if hr_stats.get('max_hr_overall') else '-')

    with tab4:
        st.markdown('<p class="section-title">❤️ 心率区间分析</p>', unsafe_allow_html=True)

        zones_def = hr.get('zones_definition', [])
        if zones_def:
            st.markdown(f'**基础设置**: 静息心率 {hr.get("resting_hr", 0)} bpm | 最大心率 {hr.get("max_hr", 0)} bpm | 心率储备 {hr.get("hrr", 0)} bpm')
            st.markdown('')

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('##### 🎯 心率区间定义')
                for z in zones_def:
                    st.markdown(
                        f'<div style="padding: 0.5rem; margin: 0.3rem 0; border-left: 4px solid {z["color"]}; background: #f8f9fa;">'
                        f'<b>{z["name"]}</b> ({int(z["min_hr"])}-{int(z["max_hr"])} bpm)<br>'
                        f'<small style="color: #666;">{z["description"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with col2:
                zone_dist = hr.get('zone_distribution', {})
                if zone_dist:
                    dist_df = pd.DataFrame([
                        {'区间': k, '次数': v['count'], '占比(%)': v['percentage'], '颜色': v['color']}
                        for k, v in zone_dist.items()
                    ])
                    if not dist_df.empty:
                        fig = px.bar(dist_df, x='区间', y='占比(%)',
                                     title='心率区间分布 (%)',
                                     color='区间',
                                     color_discrete_map={row['区间']: row['颜色'] for _, row in dist_df.iterrows()})
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown('')
        balance = hr.get('training_balance', {})
        st.markdown('##### ⚖️ 训练平衡评估')

        if balance:
            c1, c2, c3 = st.columns(3)
            c1.metric('低强度占比', f"{balance.get('easy_pct', 0):.1f}%")
            c2.metric('阈值占比', f"{balance.get('moderate_pct', 0):.1f}%")
            c3.metric('高强度占比', f"{balance.get('hard_pct', 0):.1f}%")

            st.markdown(f'**{balance.get("assessment", "")}** (评分: **{balance.get("score", 0)}/100**)')
            for rec in balance.get('recommendations', []):
                st.markdown(f'- 💡 {rec}')

    with tab5:
        st.markdown('<p class="section-title">💤 恢复状态与提醒</p>', unsafe_allow_html=True)

        reminders = recovery.get('reminders', [])
        if reminders:
            for r in reminders:
                icon = {'success': '✅', 'info': 'ℹ️', 'warning': '⚠️', 'danger': '🚨'}.get(r.get('type'), '📌')
                color = {'success': 'green', 'info': 'blue', 'warning': 'orange', 'danger': 'red'}.get(r.get('type'), 'gray')
                st.markdown(
                    f'<div style="padding: 0.8rem; margin: 0.5rem 0; border-radius: 0.4rem; '
                    f'background: {color}15; border-left: 4px solid {color};">'
                    f'<b>{icon} {r.get("title", "")}</b><br>'
                    f'<small>{r.get("message", "")}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown('')
        rec_analysis = recovery.get('recovery_analysis', {})
        if rec_analysis:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('7天训练负荷', f"{rec_analysis.get('7day_total_load', 0):.0f}")
            c2.metric('7天跑量 (km)', f"{rec_analysis.get('7day_total_distance_km', 0):.1f}")
            c3.metric('恢复评分', f"{rec_analysis.get('recovery_score', 0)}/100")
            c4.metric('恢复状态', rec_analysis.get('recovery_level', ''))

            st.markdown('')
            st.markdown('##### 📝 恢复详情')
            for d in rec_analysis.get('recovery_details', []):
                st.markdown(f'- {d}')

        st.markdown('')
        rest_analysis = recovery.get('rest_day_analysis', {})
        if rest_analysis:
            st.markdown('##### 📅 休息日情况')
            c1, c2, c3 = st.columns(3)
            c1.metric('本周已休息天数', rest_analysis.get('rest_days_count_this_week', 0))
            c2.metric('连续跑步天数', rest_analysis.get('consecutive_days_running', 0))
            c3.metric('连续运动天数', rest_analysis.get('consecutive_days_any', 0))

        overtraining = recovery.get('overtraining_risk', {})
        if overtraining:
            st.markdown('')
            risk_color = {'高风险': 'red', '中风险': 'orange', '低风险': 'yellow', '无明显风险': 'green', '数据不足': 'gray'}
            color = risk_color.get(overtraining.get('risk_level', ''), 'gray')
            st.markdown(
                f'<div style="padding: 1rem; border-radius: 0.4rem; background: {color}15; border-left: 4px solid {color};">'
                f'<b>过度训练风险: {overtraining.get("risk_level", "")}</b> (评分: {overtraining.get("risk_score", 0)})'
                f'</div>',
                unsafe_allow_html=True
            )
            for ind in overtraining.get('indicators', []):
                st.markdown(f'- {ind}')

    with tab6:
        st.markdown('<p class="section-title">🎯 训练目标追踪</p>', unsafe_allow_html=True)

        weekly = goals.get('weekly_goal', {})
        monthly = goals.get('monthly_goal', {})
        yearly = goals.get('yearly_goal', {})

        if weekly:
            st.markdown('##### 📆 本周目标')
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                progress_pct = min(100, weekly.get('progress_pct', 0))
                color = '#2ecc71' if weekly.get('on_track', True) else '#e74c3c'
                st.markdown(
                    f'<div style="padding: 1rem; background: #f8f9fa; border-radius: 0.4rem;">'
                    f'<div style="display: flex; justify-content: space-between;">'
                    f'<span>已完成 <b>{weekly.get("distance_so_far_km", 0):.1f}</b> / {weekly.get("goal_km", 0):.0f} km</span>'
                    f'<span style="color: {color}; font-weight: bold;">{weekly.get("pace_status", "")} ({progress_pct:.1f}%)</span>'
                    f'</div>'
                    f'<div style="background: #e0e0e0; height: 12px; border-radius: 6px; margin-top: 0.5rem; overflow: hidden;">'
                    f'<div style="background: {color}; height: 100%; width: {progress_pct}%; border-radius: 6px; transition: width 0.5s;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.metric('剩余 (km)', f"{weekly.get('remaining_km', 0):.1f}")
            with c3:
                st.metric('剩余天数', weekly.get('days_left', 0))
            st.caption(f'建议每天跑: {weekly.get("avg_needed_per_day_km", 0):.1f} km')

        if monthly:
            st.markdown('')
            st.markdown('##### 🗓️ 本月目标')
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                progress_pct = min(100, monthly.get('progress_pct', 0))
                color = '#3498db' if monthly.get('on_track', True) else '#e67e22'
                st.markdown(
                    f'<div style="padding: 1rem; background: #f8f9fa; border-radius: 0.4rem;">'
                    f'<div style="display: flex; justify-content: space-between;">'
                    f'<span>已完成 <b>{monthly.get("distance_so_far_km", 0):.1f}</b> / {monthly.get("goal_km", 0):.0f} km ({monthly.get("runs_count", 0)}次)</span>'
                    f'<span style="color: {color}; font-weight: bold;">{monthly.get("pace_status", "")} ({progress_pct:.1f}%)</span>'
                    f'</div>'
                    f'<div style="background: #e0e0e0; height: 12px; border-radius: 6px; margin-top: 0.5rem; overflow: hidden;">'
                    f'<div style="background: {color}; height: 100%; width: {progress_pct}%; border-radius: 6px; transition: width 0.5s;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.metric('平均配速', monthly.get('avg_pace', '--'))
            with c3:
                st.metric('累计爬升', f"{monthly.get('total_elevation_m', 0):.0f}m")

            past = monthly.get('past_3_months', [])
            if past:
                st.markdown('')
                st.markdown('##### 📈 近3个月对比')
                past_df = pd.DataFrame(past)
                past_df.columns = ['月份', '跑量(km)', '跑步次数']
                st.dataframe(past_df, use_container_width=True, hide_index=True)

        if yearly:
            st.markdown('')
            st.markdown('##### 📅 年度目标')
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                progress_pct = min(100, yearly.get('progress_pct', 0))
                color = '#9b59b6' if yearly.get('on_track', True) else '#e74c3c'
                st.markdown(
                    f'<div style="padding: 1rem; background: #f8f9fa; border-radius: 0.4rem;">'
                    f'<div style="display: flex; justify-content: space-between;">'
                    f'<span>已完成 <b>{yearly.get("distance_so_far_km", 0):.0f}</b> / {yearly.get("goal_km", 0):.0f} km ({yearly.get("runs_count", 0)}次)</span>'
                    f'<span style="color: {color}; font-weight: bold;">{yearly.get("pace_status", "")} ({progress_pct:.1f}%)</span>'
                    f'</div>'
                    f'<div style="background: #e0e0e0; height: 12px; border-radius: 6px; margin-top: 0.5rem; overflow: hidden;">'
                    f'<div style="background: {color}; height: 100%; width: {progress_pct}%; border-radius: 6px; transition: width 0.5s;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.metric('已过天数', yearly.get('days_passed', 0))
            with c3:
                st.metric('剩余天数', yearly.get('days_left', 0))

        streaks = goals.get('streaks', {})
        if streaks:
            st.markdown('')
            c1, c2, c3 = st.columns(3)
            c1.metric('🔥 当前连续跑步天数', streaks.get('current_streak_days', 0))
            c2.metric('🏆 最长连续天数', streaks.get('longest_streak_days', 0))
            c3.metric('📆 本周跑步天数', streaks.get('current_week_run_days', 0))

        trend = goals.get('trend', {})
        if trend:
            st.markdown('')
            st.markdown('##### 📊 进步趋势')
            st.write(f"里程变化: **{trend.get('status', '')}** ({trend.get('distance_change_pct', 0):+.1f}%)")
            st.write(f"配速变化: **{trend.get('pace_trend', '')}** ({trend.get('pace_change_pct', 0):+.1f}%)")

    with tab7:
        st.markdown('<p class="section-title">📝 本周训练周报</p>', unsafe_allow_html=True)

        text_report = report.get('text_report', '')
        if text_report:
            st.code(text_report, language='text')

        charts = report.get('charts', {})
        if charts:
            st.markdown('### 📈 图表分析')

            col1, col2 = st.columns(2)
            with col1:
                if charts.get('daily_distance') and not charts['daily_distance'].data:
                    pass
                elif charts.get('daily_distance'):
                    st.plotly_chart(charts['daily_distance'], use_container_width=True)
            with col2:
                if charts.get('activity_type_pie') and not charts['activity_type_pie'].data:
                    pass
                elif charts.get('activity_type_pie'):
                    st.plotly_chart(charts['activity_type_pie'], use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if charts.get('weekly_trend') and not charts['weekly_trend'].data:
                    st.info('暂无足够数据显示周趋势')
                elif charts.get('weekly_trend'):
                    st.plotly_chart(charts['weekly_trend'], use_container_width=True)
            with col4:
                if charts.get('training_load') and not charts['training_load'].data:
                    st.info('暂无训练负荷数据')
                elif charts.get('training_load'):
                    st.plotly_chart(charts['training_load'], use_container_width=True)

            if charts.get('pace_hr_scatter') and not charts['pace_hr_scatter'].data:
                pass
            elif charts.get('pace_hr_scatter'):
                st.plotly_chart(charts['pace_hr_scatter'], use_container_width=True)

        st.markdown('')
        st.markdown('### ⚠️ 异常检测')
        anomalies = report.get('anomalies', {})

        anomaly_types = [
            ('excessive_volume', '📦 训练过量', '本周训练量是否异常偏高'),
            ('low_intensity_stagnation', '🐢 长期低强度', '近期是否长期低强度训练'),
            ('performance_plateau', '📉 进步停滞', '是否进入训练平台期'),
            ('irregular_heart_rate', '💓 心率异常', '本周心率波动是否过大')
        ]

        for key, title, desc in anomaly_types:
            data = anomalies.get(key, {})
            has_issue = data.get('has_issue', False)
            icon = '❌' if has_issue else '✅'
            with st.expander(f'{icon} {title} - {desc}', expanded=has_issue):
                if has_issue:
                    for issue in data.get('issues', []):
                        st.warning(issue)
                else:
                    st.success('该指标正常')

else:
    st.info('👈 请从左侧选择"使用示例数据"或上传运动文件，然后点击"运行分析"按钮开始')

    st.markdown('')
    st.markdown('### 📁 支持的文件格式')
    col1, col2, col3 = st.columns(3)
    col1.markdown('- **GPX**: GPS Exchange Format')
    col2.markdown('- **TCX**: Training Center XML')
    col3.markdown('- **FIT**: Flexible and Interoperable Data Transfer')
    col1.markdown('- **CSV**: 批量记录表格')
    col2.markdown('- **JSON**: 结构化数据')
    col3.markdown('- 示例数据: 点击一键生成')

    st.markdown('')
    st.markdown('### 📋 CSV 格式参考')
    sample_csv = '''date,sport_type,distance_km,duration_min,avg_pace_min_km,avg_hr,max_hr,elevation_gain_m,calories,sleep_hours,injury,notes
2026-06-01 07:00:00,running,8.5,52,6.12,145,168,85,520,7.2,,轻松跑
2026-06-02 18:30:00,strength,0,45,,120,155,0,280,6.8,,力量训练'''
    st.code(sample_csv, language='csv')
