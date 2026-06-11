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
    .stDataFrame {
        font-size: 0.9rem;
    }
    .preview-section {
        background: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if 'stage' not in st.session_state:
    st.session_state.stage = 'idle'
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = pd.DataFrame()
if 'preview_df' not in st.session_state:
    st.session_state.preview_df = pd.DataFrame()
if 'preview_issues' not in st.session_state:
    st.session_state.preview_issues = pd.DataFrame()
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
if 'raw_csv_df' not in st.session_state:
    st.session_state.raw_csv_df = pd.DataFrame()
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'anomaly_mask' not in st.session_state:
    st.session_state.anomaly_mask = pd.Series()
if 'trend_weeks' not in st.session_state:
    st.session_state.trend_weeks = 8
if 'user_config' not in st.session_state:
    st.session_state.user_config = {
        'age': 35,
        'resting_hr': 60,
        'max_hr': 185,
        'monthly_goal_km': 150,
        'weekly_goal_km': 40,
        'yearly_goal_km': 1800
    }

SPORT_OPTIONS = {
    'running': '🏃 跑步',
    'cycling': '🚴 骑行',
    'strength': '💪 力量训练',
    'walking': '🚶 步行',
    'swimming': '🏊 游泳',
    'other': '📋 其他'
}


def load_data_to_preview(raw_df: pd.DataFrame):
    cleaner = ActivityCleaner()
    preview_df, issues_df = cleaner.clean_data(raw_df)
    preview_df, outlier_issues = cleaner.detect_outliers_for_preview(preview_df)
    if outlier_issues:
        outlier_df = pd.DataFrame(outlier_issues)
        if not issues_df.empty:
            issues_df = pd.concat([issues_df, outlier_df], ignore_index=True)
        else:
            issues_df = outlier_df
    st.session_state.preview_df = preview_df
    st.session_state.preview_issues = issues_df
    st.session_state.anomaly_mask = preview_df.get('is_outlier', pd.Series([False] * len(preview_df)))
    st.session_state.stage = 'preview'


def _safe_create_analyzer(analyzer_class, **kwargs):
    try:
        return analyzer_class(**kwargs)
    except TypeError as e:
        if 'unexpected keyword argument' in str(e):
            import inspect
            sig = inspect.signature(analyzer_class.__init__)
            valid_params = list(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return analyzer_class(**filtered_kwargs)
        raise


def run_analysis_pipeline():
    cfg = st.session_state.user_config

    if st.session_state.preview_df.empty:
        st.warning('请先导入数据')
        return

    clean_df = st.session_state.preview_df.copy()
    issues_df = st.session_state.preview_issues.copy()

    if 'is_outlier' in clean_df.columns:
        outlier_count = clean_df['is_outlier'].sum()
        if outlier_count > 0:
            clean_df = clean_df[~clean_df['is_outlier']].reset_index(drop=True)
            st.info(f'已排除 {outlier_count} 条异常记录，不纳入分析')
        clean_df = clean_df.drop(columns=['is_outlier'])

    if clean_df.empty:
        st.warning('排除异常记录后没有可分析的数据')
        return

    with st.spinner('正在分析配速和训练负荷...'):
        pace_analyzer = _safe_create_analyzer(
            PaceAnalyzer,
            resting_hr=cfg['resting_hr'],
            max_hr=cfg['max_hr'],
            age=cfg['age']
        )
        clean_df, pace_result = pace_analyzer.analyze(clean_df)
        st.session_state.pace_analysis = pace_result

    with st.spinner('正在分析心率区间...'):
        hr_zones = _safe_create_analyzer(
            HeartRateZones,
            resting_hr=cfg['resting_hr'],
            max_hr=cfg['max_hr'],
            age=cfg['age']
        )
        clean_df, hr_result = hr_zones.analyze(clean_df)
        st.session_state.hr_analysis = hr_result

    with st.spinner('正在评估恢复状态...'):
        recovery = RecoveryReminder()
        clean_df, recovery_result = recovery.analyze(clean_df)
        st.session_state.recovery_analysis = recovery_result

    with st.spinner('正在追踪训练目标...'):
        goals = GoalTracker(
            monthly_distance_goal_km=cfg['monthly_goal_km'],
            weekly_distance_goal_km=cfg['weekly_goal_km'],
            yearly_distance_goal_km=cfg['yearly_goal_km']
        )
        goal_result = goals.analyze(clean_df)
        st.session_state.goal_analysis = goal_result

    with st.spinner('正在生成周报...'):
        reporter = WeeklyReport()
        report_result = reporter.generate(
            clean_df,
            recovery_analysis=recovery_result,
            goal_analysis=goal_result,
            trend_weeks=st.session_state.trend_weeks
        )
        report_result['summary']['recovery_combo'] = report_result.get('recovery_combo', {})
        st.session_state.weekly_report = report_result

    st.session_state.clean_df = clean_df
    st.session_state.clean_issues = issues_df
    st.session_state.stage = 'analyzed'
    st.success('✅ 分析完成！')


def format_pace(pace_min_km):
    if pace_min_km is None or pd.isna(pace_min_km) or pace_min_km <= 0:
        return '--:--'
    total_seconds = int(round(pace_min_km * 60))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f'{minutes:02d}:{seconds:02d}'


def import_files_from_upload(uploaded_files):
    importer = DataImporter()
    temp_files = []
    csv_files = []
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            temp_files.append(tmp.name)
            if suffix.lower() == '.csv':
                csv_files.append((f.name, tmp.name))

    dfs = []
    for tf in temp_files:
        if not tf.lower().endswith('.csv'):
            single_df = importer.import_files([tf])
            if not single_df.empty:
                dfs.append(single_df)

    if csv_files:
        combined_csv = pd.DataFrame()
        for fname, tf in csv_files:
            raw_df = importer.read_csv_raw(tf)
            if raw_df is not None and not raw_df.empty:
                raw_df['_source_file'] = fname
                combined_csv = pd.concat([combined_csv, raw_df], ignore_index=True)

        if not combined_csv.empty:
            st.session_state.raw_csv_df = combined_csv
            auto_mapping = importer.auto_detect_mapping(list(combined_csv.columns))
            st.session_state.column_mapping = auto_mapping

            std_cols = ['date', 'sport_type', 'distance_km', 'duration_min',
                       'elevation_gain_m', 'avg_hr', 'max_hr', 'avg_pace_min_km',
                       'calories', 'sleep_hours', 'injury', 'notes']
            has_all_std = all(col in combined_csv.columns for col in std_cols[:4])
            if has_all_std and not auto_mapping:
                mapped_df = importer.apply_column_mapping(combined_csv, {})
                if dfs:
                    non_csv = pd.concat(dfs, ignore_index=True)
                    combined = pd.concat([non_csv, mapped_df], ignore_index=True)
                else:
                    combined = mapped_df
                st.session_state.raw_df = combined
                for tf in temp_files:
                    try: os.unlink(tf)
                    except: pass
                load_data_to_preview(combined)
                return True
            else:
                st.session_state.stage = 'column_mapping'
                for tf in temp_files:
                    try: os.unlink(tf)
                    except: pass
                return True

    for tf in temp_files:
        try:
            os.unlink(tf)
        except:
            pass

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        st.session_state.raw_df = combined
        load_data_to_preview(combined)
        return True
    return False


st.markdown('<p class="main-header">🏃 跑步健康管理助手</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">训练记录整理 · 恢复状态监测 · 目标追踪 · 智能周报</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 📊 数据源')
    use_sample = st.checkbox('使用示例数据', value=False)
    uploaded_files = st.file_uploader(
        '上传运动文件 (支持 GPX/TCX/CSV/FIT/JSON)',
        type=['gpx', 'tcx', 'csv', 'fit', 'json'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button('📥 导入并预览', type='primary', use_container_width=True):
            success = import_files_from_upload(uploaded_files)
            if not success:
                st.error('未能从上传文件中解析出数据')

    if use_sample:
        if st.button('🎯 加载示例数据', use_container_width=True):
            gen = SampleDataGenerator(seed=42)
            sample_df = gen.generate(days=75)
            st.session_state.raw_df = sample_df
            load_data_to_preview(sample_df)

    st.markdown('---')
    st.markdown('## ⚙️ 个人配置')
    cfg = st.session_state.user_config
    cfg['age'] = st.number_input('年龄', min_value=10, max_value=100, value=cfg['age'])
    cfg['resting_hr'] = st.number_input('静息心率 (bpm)', min_value=30, max_value=120, value=cfg['resting_hr'])
    cfg['max_hr'] = st.number_input('最大心率 (bpm)', min_value=100, max_value=230, value=cfg['max_hr'])

    st.markdown('### 🎯 训练目标（仅跑步）')
    cfg['weekly_goal_km'] = st.number_input('周跑量目标 (km)', min_value=0.0, value=float(cfg['weekly_goal_km']), step=5.0)
    cfg['monthly_goal_km'] = st.number_input('月跑量目标 (km)', min_value=0.0, value=float(cfg['monthly_goal_km']), step=10.0)
    cfg['yearly_goal_km'] = st.number_input('年跑量目标 (km)', min_value=0.0, value=float(cfg['yearly_goal_km']), step=100.0)

    st.markdown('---')

    if st.session_state.stage == 'preview':
        if st.button('✅ 确认并开始分析', type='primary', use_container_width=True):
            run_analysis_pipeline()
    elif st.session_state.stage == 'analyzed':
        if st.button('🔄 重新分析', use_container_width=True):
            run_analysis_pipeline()

if st.session_state.stage == 'idle':
    st.info('👈 请从左侧选择"使用示例数据"或上传运动文件，先预览数据再开始分析')

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

elif st.session_state.stage == 'column_mapping':
    st.markdown('<p class="section-title">🔗 CSV 列名匹配</p>', unsafe_allow_html=True)

    raw_df = st.session_state.raw_csv_df
    importer = DataImporter()

    st.info('请将 CSV 文件中的列名与标准字段对应起来。日期为必填项，其他字段可选。系统已自动尝试匹配，您可以手动调整。')

    st.markdown('')
    st.markdown('#### 📝 数据预览（前5行）')
    st.dataframe(raw_df.head(), use_container_width=True, hide_index=True)

    st.markdown('')
    st.markdown('#### 🔄 列名匹配')

    csv_cols = list(raw_df.columns)
    if '_source_file' in csv_cols:
        csv_cols.remove('_source_file')
    csv_options = [''] + csv_cols

    col1, col2 = st.columns(2)
    new_mapping = {}

    for i, field in enumerate(importer.standard_fields):
        key = field['key']
        label = field['label']
        required = field['required']

        curr_value = st.session_state.column_mapping.get(key, '')
        if curr_value not in csv_options:
            curr_value = ''

        with col1 if i % 2 == 0 else col2:
            label_text = f'{label} {"*" if required else ""}'
            selected = st.selectbox(
                label_text,
                options=csv_options,
                index=csv_options.index(curr_value) if curr_value in csv_options else 0,
                key=f'map_{key}'
            )
            if selected:
                new_mapping[key] = selected

    st.markdown('')
    col_left, col_right = st.columns([1, 1])

    with col_left:
        if st.button('↩️ 重新自动匹配', use_container_width=True):
            new_mapping = importer.auto_detect_mapping(list(raw_df.columns))
            st.session_state.column_mapping = new_mapping
            st.rerun()

    with col_right:
        if st.button('✅ 确认匹配并继续 →', type='primary', use_container_width=True):
            if not new_mapping.get('date'):
                st.error('请至少匹配"日期/开始时间"字段')
            else:
                mapped_df = importer.apply_column_mapping(raw_df, new_mapping)
                st.session_state.raw_df = mapped_df
                st.session_state.column_mapping = new_mapping
                load_data_to_preview(mapped_df)
                st.rerun()

elif st.session_state.stage == 'preview':
    st.markdown('<p class="section-title">👀 数据导入预览</p>', unsafe_allow_html=True)

    preview_df = st.session_state.preview_df
    preview_issues = st.session_state.preview_issues

    outlier_count = preview_df['is_outlier'].sum() if 'is_outlier' in preview_df.columns else 0
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('原始记录数', len(st.session_state.raw_df))
    col2.metric('清洗后记录数', len(preview_df))
    col3.metric('运动类型数', preview_df['sport_type'].nunique() if len(preview_df) > 0 else 0)
    col4.metric('发现问题数', len(preview_issues))
    col5.metric('异常记录数', outlier_count, delta=f'-{outlier_count} 条待处理')

    st.markdown('')
    st.markdown('#### 🏷️ 运动类型分布')
    type_counts = preview_df['sport_type'].value_counts()
    type_labels = [SPORT_OPTIONS.get(t, t) for t in type_counts.index]
    cols = st.columns(min(len(type_counts), 6))
    for i, (sport, count) in enumerate(zip(type_counts.index, type_counts.values)):
        cols[i].metric(SPORT_OPTIONS.get(sport, sport), count)

    st.markdown('')
    st.markdown('#### ✏️ 手动校正（可选）')
    st.caption('可以在这里批量修改运动类型、距离、时长、伤痛和睡眠，确认后再进入分析')

    edit_cols = ['date', 'sport_type', 'distance_km', 'duration_min',
                 'avg_hr', 'sleep_hours', 'injury', 'notes']
    display_df = preview_df[edit_cols].copy()

    display_df['sport_type'] = display_df['sport_type'].map(
        lambda x: SPORT_OPTIONS.get(x, x)
    )

    with st.expander('📝 展开编辑数据', expanded=False):
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows='fixed',
            column_config={
                'date': st.column_config.TextColumn('日期', disabled=True),
                'sport_type': st.column_config.SelectboxColumn(
                    '运动类型',
                    options=list(SPORT_OPTIONS.values()),
                    required=True
                ),
                'distance_km': st.column_config.NumberColumn('距离(km)', min_value=0, step=0.1),
                'duration_min': st.column_config.NumberColumn('时长(min)', min_value=0, step=1),
                'avg_hr': st.column_config.NumberColumn('平均心率', min_value=0, step=1),
                'sleep_hours': st.column_config.NumberColumn('睡眠(h)', min_value=0, step=0.5),
                'injury': st.column_config.TextColumn('伤痛'),
                'notes': st.column_config.TextColumn('备注'),
            },
            height=400,
            hide_index=True
        )

        if st.button('💾 保存修改'):
            sport_reverse = {v: k for k, v in SPORT_OPTIONS.items()}
            edited_df['sport_type'] = edited_df['sport_type'].map(
                lambda x: sport_reverse.get(x, x)
            )

            for col in edit_cols:
                if col in edited_df.columns:
                    preview_df[col] = edited_df[col].values

            pace_col = preview_df.get('avg_pace_min_km')
            preview_df['avg_pace_min_km'] = np.where(
                (preview_df['distance_km'] > 0) & (preview_df['duration_min'] > 0),
                preview_df['duration_min'] / preview_df['distance_km'],
                preview_df.get('avg_pace_min_km')
            )

            st.session_state.preview_df = preview_df
            st.success('修改已保存！')

    outlier_count = preview_df['is_outlier'].sum() if 'is_outlier' in preview_df.columns else 0
    if outlier_count > 0:
        st.markdown('')
        st.markdown(f'#### ⚠️ 异常值检测（发现 {outlier_count} 条异常记录）')
        st.caption('以下记录的数值可能存在异常，您可以手动修正或排除这些记录')

        with st.expander('🔍 查看并修正异常值', expanded=True):
            outlier_df = preview_df[preview_df['is_outlier']].copy()
            outlier_edit_cols = ['date', 'sport_type', 'distance_km', 'duration_min',
                                 'avg_hr', 'max_hr', 'avg_pace_min_km', 'notes']
            outlier_display = outlier_df[outlier_edit_cols].copy()

            outlier_display['sport_type'] = outlier_display['sport_type'].map(
                lambda x: SPORT_OPTIONS.get(x, x)
            )

            outlier_issues = preview_issues[preview_issues['issue_type'] == '数值异常']
            if not outlier_issues.empty:
                st.markdown('##### ❗ 异常详情')
                for _, row in outlier_issues.iterrows():
                    st.warning(f"**{row['date']}** ({SPORT_OPTIONS.get(row['sport_type'], row['sport_type'])}): {row['details']}")

            st.markdown('##### ✏️ 修正异常值')
            edited_outliers = st.data_editor(
                outlier_display,
                use_container_width=True,
                num_rows='fixed',
                column_config={
                    'date': st.column_config.TextColumn('日期', disabled=True),
                    'sport_type': st.column_config.SelectboxColumn(
                        '运动类型',
                        options=list(SPORT_OPTIONS.values()),
                        required=True
                    ),
                    'distance_km': st.column_config.NumberColumn('距离(km)', min_value=0, step=0.1),
                    'duration_min': st.column_config.NumberColumn('时长(min)', min_value=0, step=1),
                    'avg_hr': st.column_config.NumberColumn('平均心率', min_value=0, max_value=240, step=1),
                    'max_hr': st.column_config.NumberColumn('最大心率', min_value=0, max_value=250, step=1),
                    'avg_pace_min_km': st.column_config.NumberColumn('配速(min/km)', min_value=0, step=0.1),
                    'notes': st.column_config.TextColumn('备注'),
                },
                height=300,
                hide_index=True
            )

            col_fix1, col_fix2, col_fix3 = st.columns(3)
            with col_fix1:
                if st.button('💾 保存异常值修正', use_container_width=True):
                    sport_reverse = {v: k for k, v in SPORT_OPTIONS.items()}
                    edited_outliers['sport_type'] = edited_outliers['sport_type'].map(
                        lambda x: sport_reverse.get(x, x)
                    )
                    for idx in outlier_df.index:
                        for col in outlier_edit_cols:
                            if idx in edited_outliers.index:
                                preview_df.loc[idx, col] = edited_outliers.loc[idx, col]
                    preview_df.loc[outlier_df.index, 'is_outlier'] = False
                    preview_df['avg_pace_min_km'] = np.where(
                        (preview_df['distance_km'] > 0) & (preview_df['duration_min'] > 0),
                        preview_df['duration_min'] / preview_df['distance_km'],
                        preview_df.get('avg_pace_min_km')
                    )
                    st.session_state.preview_df = preview_df
                    st.success('修正已保存！已重新检测异常值')
                    cleaner = ActivityCleaner()
                    preview_df, new_outlier_issues = cleaner.detect_outliers_for_preview(preview_df)
                    st.session_state.preview_df = preview_df
                    st.rerun()

            with col_fix2:
                if st.button('🚫 排除所有异常记录', use_container_width=True):
                    preview_df = preview_df[~preview_df['is_outlier']].reset_index(drop=True)
                    st.session_state.preview_df = preview_df
                    st.success(f'已排除 {outlier_count} 条异常记录')
                    st.rerun()

            with col_fix3:
                if st.button('✅ 保留（已确认无误）', use_container_width=True):
                    preview_df.loc[outlier_df.index, 'is_outlier'] = False
                    st.session_state.preview_df = preview_df
                    st.success('已确认保留这些记录')
                    st.rerun()

    st.markdown('')
    with st.expander('🧹 查看清洗日志', expanded=False):
        if not preview_issues.empty:
            severity_map = {'info': 'ℹ️ 信息', 'warning': '⚠️ 警告', 'error': '❌ 错误'}
            display_issues = preview_issues.copy()
            display_issues['severity'] = display_issues['severity'].map(severity_map).fillna(display_issues['severity'])
            st.dataframe(display_issues, use_container_width=True, hide_index=True)
        else:
            st.info('✅ 数据质量良好，未发现需要修正的问题')

    st.markdown('')
    st.markdown('---')
    col_left, col_right = st.columns([1, 1])

    with col_left:
        remaining_outliers = preview_df['is_outlier'].sum() if 'is_outlier' in preview_df.columns else 0
        if remaining_outliers > 0:
            st.warning(f'还有 {remaining_outliers} 条异常记录未处理')
        else:
            st.success('✅ 所有异常记录已处理')

    with col_right:
        if st.button('✅ 确认数据并开始分析 →', type='primary', use_container_width=True):
            run_analysis_pipeline()

elif st.session_state.stage == 'analyzed':
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

        all_summary = pace.get('all_sports_summary', {})
        running_summary = pace.get('summary', {})
        by_sport = pace.get('by_sport', {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric('总活动次数', all_summary.get('total_activities', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card metric-card-green">', unsafe_allow_html=True)
            st.metric('总时长 (小时)', f"{all_summary.get('total_duration_h', 0):.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card metric-card-orange">', unsafe_allow_html=True)
            st.metric('总训练负荷', f"{all_summary.get('total_load', 0):.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card metric-card-blue">', unsafe_allow_html=True)
            st.metric('运动类型数', all_summary.get('sport_count', 0))
            st.markdown('</div>', unsafe_allow_html=True)

        if running_summary.get('has_data', False):
            st.markdown('')
            st.markdown('##### 🏃 跑步专项')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('跑步次数', running_summary.get('total_runs', 0))
            c2.metric('跑步距离 (km)', f"{running_summary.get('total_distance_km', 0):.1f}")
            c3.metric('平均配速', running_summary.get('avg_pace', '--:--'))
            c4.metric('最佳配速', running_summary.get('best_pace', '--:--'))

        if by_sport:
            st.markdown('')
            st.markdown('##### 🏅 各运动类型明细')
            sport_rows = []
            for key, data in by_sport.items():
                sport_rows.append({
                    '运动类型': data.get('name', key),
                    '次数': data.get('count', 0),
                    '距离(km)': data.get('total_distance_km', 0),
                    '时长(h)': data.get('total_duration_h', 0),
                    '平均配速': data.get('avg_pace', '--'),
                    '平均心率': data.get('avg_hr', '-'),
                    '训练负荷': data.get('total_load', 0),
                    '爬升(m)': data.get('total_elevation_m', 0),
                })
            sport_df = pd.DataFrame(sport_rows)
            st.dataframe(sport_df, use_container_width=True, hide_index=True)

        st.markdown('')
        st.markdown('<p class="section-title">📅 最近训练记录</p>', unsafe_allow_html=True)

        display_cols = ['date', 'sport_type', 'distance_km', 'duration_min',
                        'avg_pace_min_km', 'avg_hr', 'elevation_gain_m', 'training_load',
                        'sleep_hours', 'injury', 'notes']
        available_cols = [c for c in display_cols if c in clean_df.columns]
        display_df = clean_df[available_cols].copy()

        column_name_map = {
            'date': '日期', 'sport_type': '运动类型', 'distance_km': '距离(km)',
            'duration_min': '时长(min)', 'avg_pace_min_km': '配速(min/km)',
            'avg_hr': '平均心率', 'elevation_gain_m': '爬升(m)',
            'training_load': '训练负荷', 'sleep_hours': '睡眠(h)',
            'injury': '伤痛', 'notes': '备注'
        }
        display_df.columns = [column_name_map.get(c, c) for c in available_cols]

        if '运动类型' in display_df.columns:
            display_df['运动类型'] = display_df['运动类型'].map(SPORT_OPTIONS).fillna(display_df['运动类型'])
        if '配速(min/km)' in display_df.columns:
            display_df['配速(min/km)'] = display_df['配速(min/km)'].apply(
                lambda x: format_pace(x) if pd.notna(x) else '--'
            )

        st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
        st.caption(f'共 {len(clean_df)} 条记录，显示最近 20 条')

    with tab2:
        st.markdown('<p class="section-title">🧹 数据清洗日志</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric('原始记录数', len(st.session_state.raw_df))
        col2.metric('清洗后记录数', len(clean_df))
        col3.metric('发现问题数', len(st.session_state.clean_issues))

        if not st.session_state.clean_issues.empty:
            issues_df = st.session_state.clean_issues.copy()
            severity_map = {'info': 'ℹ️ 信息', 'warning': '⚠️ 警告', 'error': '❌ 错误'}
            issues_df['severity'] = issues_df['severity'].map(severity_map).fillna(issues_df['severity'])

            st.dataframe(issues_df, use_container_width=True, hide_index=True)

            st.markdown('')
            st.markdown('##### 📊 问题类型统计')
            type_counts = issues_df['issue_type'].value_counts().reset_index()
            type_counts.columns = ['问题类型', '数量']
            st.bar_chart(type_counts.set_index('问题类型'))
        else:
            st.info('✅ 数据质量良好，未发现需要修正的问题')

    with tab3:
        st.markdown('<p class="section-title">⏱️ 配速与负荷分析</p>', unsafe_allow_html=True)

        running_summary = pace.get('summary', {})
        pace_dist = pace.get('pace_distribution', {})
        elev_stats = pace.get('elevation_stats', {})
        hr_stats = pace.get('heart_rate_stats', {})

        if running_summary.get('has_data', False):
            col1, col2 = st.columns(2)

            with col1:
                if pace_dist.get('has_data', False):
                    dist_df = pd.DataFrame([
                        {'配速区间': k, '次数': v}
                        for k, v in pace_dist.get('distribution', {}).items() if v > 0
                    ])
                    if not dist_df.empty:
                        import plotly.express as px
                        fig = px.bar(dist_df, x='配速区间', y='次数',
                                     title='🏃 跑步配速分布', color='次数',
                                     color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown('##### 📊 配速统计')
                st.write(f"中位数配速: **{pace_dist.get('median_pace', '--')}**")
                st.write(f"Q25: {pace_dist.get('pace_q25', '--')} / Q75: {pace_dist.get('pace_q75', '--')}")
                st.write(f"配速标准差: {pace_dist.get('pace_std', 0):.2f} min/km")

            with col2:
                if elev_stats.get('has_data', False):
                    terrain_df = pd.DataFrame([
                        {'地形类型': k, '次数': v}
                        for k, v in elev_stats.get('terrain_distribution', {}).items() if v > 0
                    ])
                    if not terrain_df.empty:
                        fig = px.pie(terrain_df, names='地形类型', values='次数',
                                     title='🏔️ 训练地形分布', hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown('##### 🏔️ 爬升统计')
                st.write(f"平均爬升: **{elev_stats.get('avg_elevation_m', 0):.0f} m**")
                st.write(f"最大爬升: **{elev_stats.get('max_elevation_m', 0):.0f} m**")
        else:
            st.info('暂无跑步配速数据（可能只有骑行、力量等其他运动类型）')

        st.markdown('')
        st.markdown('##### ❤️ 心率概况（所有运动）')
        if hr_stats.get('has_data', False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('平均心率', f"{hr_stats.get('avg_hr_overall', 0):.0f} bpm")
            c2.metric('最低平均心率', f"{hr_stats.get('avg_hr_min', 0):.0f} bpm")
            c3.metric('最高平均心率', f"{hr_stats.get('avg_hr_max', 0):.0f} bpm")
            c4.metric('有心率记录', f"{hr_stats.get('hr_count', 0)} 次")
        else:
            st.info('暂无心率数据')

    with tab4:
        st.markdown('<p class="section-title">❤️ 心率区间分析</p>', unsafe_allow_html=True)

        zones_def = hr.get('zones_definition', [])
        has_hr_data = hr.get('has_hr_data', False)

        st.markdown(f'**基础设置**: 静息心率 {hr.get("resting_hr", 0)} bpm | 最大心率 {hr.get("max_hr", 0)} bpm | 心率储备 {hr.get("hrr", 0)} bpm')
        st.caption(f"共有 {hr.get('total_with_hr', 0)} 次训练有心率数据，{hr.get('total_without_hr', 0)} 次无心率数据")
        st.markdown('')

        if zones_def:
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
                        for k, v in zone_dist.items() if v.get('count', 0) > 0
                    ])
                    if not dist_df.empty:
                        import plotly.express as px
                        fig = px.bar(dist_df, x='区间', y='占比(%)',
                                     title='心率区间分布 (%)',
                                     color='区间',
                                     color_discrete_map={row['区间']: row['颜色'] for _, row in dist_df.iterrows()})
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown('')
        balance = hr.get('training_balance', {})
        st.markdown('##### ⚖️ 训练平衡评估')

        if balance.get('has_data', False):
            c1, c2, c3 = st.columns(3)
            c1.metric('低强度占比', f"{balance.get('easy_pct', 0):.1f}%")
            c2.metric('阈值占比', f"{balance.get('moderate_pct', 0):.1f}%")
            c3.metric('高强度占比', f"{balance.get('hard_pct', 0):.1f}%")

            st.markdown(f'**{balance.get("assessment", "")}** (评分: **{balance.get("score", 0)}/100**)')
            for rec in balance.get('recommendations', []):
                st.markdown(f'- 💡 {rec}')
        else:
            st.info(balance.get('assessment', '暂无足够心率数据进行训练平衡评估'))

        by_sport_hr = hr.get('by_sport', {})
        if by_sport_hr:
            st.markdown('')
            st.markdown('##### 🏅 各运动类型心率统计')
            rows = []
            for key, data in by_sport_hr.items():
                rows.append({
                    '运动类型': data.get('name', key),
                    '总次数': data.get('total_count', 0),
                    '有心率次数': data.get('with_hr_count', 0),
                    '平均心率': data.get('avg_hr_avg', '-'),
                    '总负荷': data.get('total_load', 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
            c1.metric('7天总负荷', f"{rec_analysis.get('7day_total_load', 0):.0f}")
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

        st.markdown('')
        st.markdown('##### 😴 睡眠与伤痛统计')
        c1, c2 = st.columns(2)
        c1.metric('有伤痛记录次数', recovery.get('injury_count', 0))
        c2.metric('睡眠异常次数', recovery.get('sleep_issues_count', 0))

    with tab6:
        st.markdown('<p class="section-title">🎯 训练目标追踪</p>', unsafe_allow_html=True)

        if not goals.get('has_running_data', True):
            st.info('暂无跑步数据，跑步目标追踪暂不可用。可以查看下方其他运动的总览。')
        else:
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

        all_sports = goals.get('all_sports_summary', {})
        if all_sports:
            st.markdown('')
            st.markdown('##### 📋 所有运动类型总览')
            rows = []
            for key, data in all_sports.items():
                rows.append({
                    '运动类型': data.get('name', key),
                    '次数': data.get('count', 0),
                    '总距离(km)': data.get('total_distance_km', 0),
                    '总时长(h)': data.get('total_duration_h', 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab7:
        week_period = report.get('week_period', '')
        week_start = report.get('week_start', '')
        week_end = report.get('week_end', '')
        title_date = week_period if week_period else '本周'

        st.markdown(f'<p class="section-title">📝 训练周报（{title_date}）</p>', unsafe_allow_html=True)

        text_report = report.get('text_report', '')
        if text_report:
            with st.expander('📄 查看文本周报', expanded=False):
                st.code(text_report, language='text')

        st.markdown('')
        col_export1, col_export2, col_export3 = st.columns(3)
        date_str = week_start.replace('-', '') if week_start else datetime.now().strftime("%Y%m%d")
        with col_export1:
            md_content = report.get('markdown_report', '')
            if md_content:
                st.download_button(
                    '📥 下载 Markdown 周报',
                    data=md_content,
                    file_name=f'训练周报_{date_str}.md',
                    mime='text/markdown',
                    use_container_width=True
                )
        with col_export2:
            export_data = report.get('export_data', pd.DataFrame())
            if not export_data.empty:
                csv_data = export_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    '📊 导出全部训练数据 (CSV)',
                    data=csv_data,
                    file_name=f'训练明细_{date_str}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        with col_export3:
            summary_export = report.get('summary_export', pd.DataFrame())
            if not summary_export.empty:
                csv_data = summary_export.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    '📋 导出周报汇总 (CSV)',
                    data=csv_data,
                    file_name=f'周报汇总_{date_str}.csv',
                    mime='text/csv',
                    use_container_width=True
                )

        st.markdown('')
        charts = report.get('charts', {})
        if charts:
            st.markdown('### 📈 图表分析')

            col1, col2 = st.columns(2)
            with col1:
                if charts.get('daily_distance'):
                    st.plotly_chart(charts['daily_distance'], use_container_width=True)
            with col2:
                if charts.get('activity_type_pie'):
                    st.plotly_chart(charts['activity_type_pie'], use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if charts.get('weekly_trend'):
                    st.plotly_chart(charts['weekly_trend'], use_container_width=True)
            with col4:
                if charts.get('training_load'):
                    st.plotly_chart(charts['training_load'], use_container_width=True)

            if charts.get('pace_hr_scatter'):
                st.plotly_chart(charts['pace_hr_scatter'], use_container_width=True)

        st.markdown('')
        st.markdown('### 📊 训练负荷与恢复组合视图')
        combo_chart = charts.get('load_recovery_combo')
        if combo_chart:
            st.plotly_chart(combo_chart, use_container_width=True)

        monthly_trend = report.get('monthly_trend', {})
        if monthly_trend.get('has_data', False):
            st.markdown('')
            trend_weeks = st.session_state.get('trend_weeks', 8)
            st.markdown(f'### 📈 月度训练趋势（近{trend_weeks}周）')

            col_trend1, col_trend2 = st.columns([3, 1])
            with col_trend1:
                trend_analysis = monthly_trend.get('trend_analysis', '')
                if trend_analysis:
                    st.info(f'**趋势分析**: {trend_analysis}')

                trend_direction = monthly_trend.get('trend_direction', '')
                if trend_direction:
                    direction_desc = {
                        'increasing': '📈 训练负荷呈上升趋势，注意循序渐进',
                        'decreasing': '📉 训练负荷呈下降趋势，可能处于恢复期',
                        'stable': '➡️ 训练负荷保持稳定，持续性良好'
                    }
                    st.success(f'**趋势判断**: {direction_desc.get(trend_direction, trend_direction)}')

                weekly_data = monthly_trend.get('weekly_data', [])
                injury_weeks_count = sum(1 for w in weekly_data if w.get('has_injury'))
                if injury_weeks_count > 0:
                    st.warning(f'**伤痛情况**: 近{trend_weeks}周中有 {injury_weeks_count} 周存在伤痛记录，建议关注恢复情况')

            with col_trend2:
                st.markdown('**趋势周期**')
                selected_weeks = st.selectbox(
                    '选择周期',
                    options=[4, 8, 12],
                    index=[4, 8, 12].index(trend_weeks) if trend_weeks in [4, 8, 12] else 1,
                    format_func=lambda x: f'近 {x} 周',
                    label_visibility='collapsed',
                    key='trend_weeks_select'
                )
                if selected_weeks != trend_weeks:
                    st.session_state.trend_weeks = selected_weeks
                    with st.spinner('正在更新趋势分析...'):
                        reporter = WeeklyReport()
                        new_report = reporter.generate(
                            st.session_state.clean_df,
                            recovery_analysis=st.session_state.recovery_analysis,
                            goal_analysis=st.session_state.goal_analysis,
                            trend_weeks=selected_weeks
                        )
                        new_report['summary']['recovery_combo'] = new_report.get('recovery_combo', {})
                        st.session_state.weekly_report = new_report
                    st.rerun()

            monthly_chart = charts.get('monthly_trend')
            if monthly_chart:
                st.plotly_chart(monthly_chart, use_container_width=True)

        recovery_combo = report.get('recovery_combo', {})
        if recovery_combo:
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            col_rec1.metric('恢复评分', f"{recovery_combo.get('recovery_score', 0)}/100")
            col_rec2.metric('7天总负荷', f"{recovery_combo.get('total_load_7d', 0):.0f}")
            col_rec3.metric('发现问题', recovery_combo.get('issue_count', 0))

            st.markdown('')
            st.markdown('##### ⚠️ 问题汇总')
            issues = recovery_combo.get('issues', [])
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success('✅ 未发现明显问题')

            st.markdown('')
            st.markdown('##### 💡 综合建议')
            recs = recovery_combo.get('recommendations', [])
            for rec in recs:
                st.markdown(f'- {rec}')

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
