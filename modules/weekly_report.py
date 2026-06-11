import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class WeeklyReport:
    def __init__(self):
        self.sport_names = {
            'running': '跑步',
            'cycling': '骑行',
            'strength': '力量训练',
            'walking': '步行',
            'swimming': '游泳',
            'other': '其他'
        }
        self.sport_colors = {
            'running': '#2ecc71',
            'cycling': '#3498db',
            'strength': '#e67e22',
            'walking': '#1abc9c',
            'swimming': '#9b59b6',
            'other': '#95a5a6'
        }

    def generate(self, df: pd.DataFrame, recovery_analysis: Optional[Dict] = None,
                 goal_analysis: Optional[Dict] = None, trend_weeks: int = 8) -> Dict:
        if df.empty:
            return {
                'week_period': '',
                'week_start': '',
                'week_end': '',
                'summary': {},
                'anomalies': {},
                'charts': {},
                'text_report': '',
                'markdown_report': '',
                'recovery_combo': {},
                'export_data': pd.DataFrame(),
                'summary_export': pd.DataFrame(),
                'monthly_trend': {},
                'trend_weeks': trend_weeks,
                'has_data': False
            }

        df = df.copy()
        df = df.sort_values('date_parsed')

        latest_date = df['date_parsed'].max()
        today = latest_date.to_pydatetime()
        week_start = today - timedelta(days=today.weekday())
        last_week_start = week_start - timedelta(days=7)
        week_end = today

        this_week_df = df[df['date_parsed'] >= week_start].copy()
        last_week_df = df[(df['date_parsed'] >= last_week_start) & (df['date_parsed'] < week_start)].copy()

        week_summary = self._calculate_week_summary(this_week_df, last_week_df)
        anomalies = self._detect_anomalies(df, this_week_df)
        recovery_combo = self._generate_recovery_combo(this_week_df, df, recovery_analysis, anomalies)
        monthly_trend = self._analyze_monthly_trend(df, weeks=trend_weeks)
        charts = self._generate_charts(this_week_df, df, monthly_trend)
        text_report = self._generate_text_report(week_summary, anomalies)
        week_period = f'{week_start.strftime("%Y-%m-%d")} ~ {week_end.strftime("%Y-%m-%d")}'
        markdown_report = self._generate_markdown_report(
            week_summary, anomalies, recovery_analysis, goal_analysis,
            recovery_combo, monthly_trend, week_period
        )
        export_data, summary_export = self._prepare_export_data(
            df, this_week_df, week_summary, recovery_combo, anomalies, monthly_trend, week_period
        )

        return {
            'week_period': week_period,
            'week_start': week_start.strftime("%Y-%m-%d"),
            'week_end': week_end.strftime("%Y-%m-%d"),
            'summary': week_summary,
            'anomalies': anomalies,
            'charts': charts,
            'text_report': text_report,
            'markdown_report': markdown_report,
            'recovery_combo': recovery_combo,
            'monthly_trend': monthly_trend,
            'export_data': export_data,
            'summary_export': summary_export,
            'trend_weeks': trend_weeks,
            'has_data': True
        }

    def _calculate_week_summary(self, this_week: pd.DataFrame, last_week: pd.DataFrame) -> Dict:
        def calc_stats(df: pd.DataFrame) -> Dict:
            stats = {
                'total_activities': len(df),
                'active_days': df['date_parsed'].dt.date.nunique() if len(df) > 0 else 0,
                'total_duration_h': round(df['duration_min'].sum() / 60, 2) if 'duration_min' in df.columns else 0,
                'total_distance_km': round(df['distance_km'].sum(), 2) if 'distance_km' in df.columns else 0,
                'total_elevation_m': round(df['elevation_gain_m'].sum(), 1) if 'elevation_gain_m' in df.columns else 0,
                'total_training_load': round(df['training_load'].sum(), 1) if 'training_load' in df.columns else 0,
                'avg_hr': None,
                'avg_sleep_h': None,
                'injury_count': int(df['is_injured'].sum()) if 'is_injured' in df.columns else 0,
                'by_sport': {}
            }

            if 'avg_hr' in df.columns and df['avg_hr'].notna().any():
                stats['avg_hr'] = round(df['avg_hr'].mean(), 1)

            if 'sleep_hours' in df.columns and df['sleep_hours'].notna().any():
                stats['avg_sleep_h'] = round(df['sleep_hours'].mean(), 1)

            for sport in df['sport_type'].unique():
                sport_df = df[df['sport_type'] == sport]
                has_pace = sport in ['running', 'cycling', 'walking']

                avg_pace_val = None
                if has_pace and 'avg_pace_min_km' in sport_df.columns and sport_df['avg_pace_min_km'].notna().any():
                    avg_pace_val = sport_df['avg_pace_min_km'].mean()

                stats['by_sport'][sport] = {
                    'name': self.sport_names.get(sport, sport),
                    'count': len(sport_df),
                    'distance_km': round(sport_df['distance_km'].sum(), 2),
                    'duration_h': round(sport_df['duration_min'].sum() / 60, 2),
                    'avg_pace_min_km': avg_pace_val,
                    'elevation_m': round(sport_df['elevation_gain_m'].sum(), 1),
                    'load': round(sport_df['training_load'].sum(), 1) if 'training_load' in sport_df.columns else 0
                }

            running = df[df['sport_type'] == 'running']
            stats['running_count'] = len(running)
            stats['running_distance_km'] = round(running['distance_km'].sum(), 2)
            stats['running_duration_h'] = round(running['duration_min'].sum() / 60, 2)

            if 'avg_pace_min_km' in running.columns and running['avg_pace_min_km'].notna().any():
                stats['avg_pace_running'] = running['avg_pace_min_km'].mean()
            else:
                stats['avg_pace_running'] = None

            return stats

        this_week_stats = calc_stats(this_week)
        last_week_stats = calc_stats(last_week)

        def calc_change(curr, prev):
            if prev and prev != 0 and isinstance(curr, (int, float)) and isinstance(prev, (int, float)):
                return round((curr - prev) / prev * 100, 1)
            return 0

        comparison = {}
        for key in ['total_duration_h', 'total_training_load', 'total_distance_km', 'running_count', 'running_distance_km']:
            curr = this_week_stats.get(key, 0)
            prev = last_week_stats.get(key, 0)
            comparison[f'{key}_change_pct'] = calc_change(curr, prev)

        return {
            'this_week': this_week_stats,
            'last_week': last_week_stats,
            'comparison': comparison
        }

    def _detect_anomalies(self, all_df: pd.DataFrame, this_week: pd.DataFrame) -> Dict:
        excessive = self._detect_excessive_volume(this_week, all_df)
        low_intensity = self._detect_low_intensity_stagnation(all_df)
        plateau = self._detect_performance_plateau(all_df)
        irregular_hr = self._detect_irregular_heart_rate(this_week)

        return {
            'excessive_volume': excessive,
            'low_intensity_stagnation': low_intensity,
            'performance_plateau': plateau,
            'irregular_heart_rate': irregular_hr
        }

    def _detect_excessive_volume(self, this_week: pd.DataFrame, all_df: pd.DataFrame) -> Dict:
        issues = []
        if this_week.empty:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        total_distance = this_week['distance_km'].sum() if 'distance_km' in this_week.columns else 0
        total_duration = this_week['duration_min'].sum() if 'duration_min' in this_week.columns else 0
        total_load = this_week['training_load'].sum() if 'training_load' in this_week.columns else 0

        active_days = this_week['date_parsed'].dt.date.nunique()

        all_active = all_df[all_df['sport_type'].isin(['running', 'cycling', 'strength'])]
        if len(all_active) >= 10:
            weekly_dists = all_active.groupby(all_active['date_parsed'].dt.isocalendar().week)['distance_km'].sum()
            if len(weekly_dists) >= 3:
                avg_weekly = weekly_dists.mean()
                std_weekly = weekly_dists.std()
                if std_weekly > 0 and total_distance > avg_weekly + 2 * std_weekly and total_distance > 60:
                    issues.append(f'本周总里程{total_distance:.1f}km显著高于历史平均（{avg_weekly:.1f}km），增加量超过2个标准差')

        if total_load > 1800:
            issues.append(f'本周训练总负荷{total_load:.0f}很高，注意充分恢复')
        elif total_load > 1200:
            issues.append(f'本周训练总负荷{total_load:.0f}偏高')

        if total_duration > 840:
            issues.append(f'本周训练总时长{total_duration/60:.1f}小时，超过14小时')

        if active_days >= 6:
            issues.append(f'本周训练{active_days}天，建议安排至少1个完全休息日')

        severity = 'high' if len(issues) >= 2 else ('medium' if len(issues) == 1 else 'low')

        return {
            'has_issue': len(issues) > 0,
            'issues': issues,
            'total_distance_km': round(total_distance, 2),
            'total_duration_h': round(total_duration / 60, 2),
            'total_load': round(total_load, 1),
            'active_days': active_days,
            'severity': severity
        }

    def _detect_low_intensity_stagnation(self, all_df: pd.DataFrame) -> Dict:
        issues = []
        if len(all_df) < 8:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        all_with_hr = all_df[all_df['avg_hr'].notna()].copy()
        all_with_hr = all_with_hr.sort_values('date_parsed')

        if len(all_with_hr) < 8:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        recent = all_with_hr.tail(8)

        max_hr = all_with_hr['max_hr'].max() if 'max_hr' in all_with_hr.columns else 180
        if max_hr and max_hr > 0:
            hr_pcts = []
            for hr in recent['avg_hr'].dropna():
                if hr and hr > 0:
                    hr_pcts.append(hr / max_hr)
            if len(hr_pcts) >= 5 and all(p < 0.65 for p in hr_pcts):
                issues.append(f'近{len(hr_pcts)}次训练平均心率均低于最大心率的65%，长期低强度训练可能限制进步')

        running = all_df[all_df['sport_type'] == 'running'].copy()
        running = running.sort_values('date_parsed')
        if len(running) >= 8 and 'avg_pace_min_km' in running.columns:
            recent_running = running.tail(8)
            paces = recent_running['avg_pace_min_km'].dropna().values
            if len(paces) >= 5:
                pace_std = np.std(paces)
                avg_pace = np.mean(paces)
                if avg_pace > 0 and pace_std / avg_pace < 0.08 and avg_pace > 7:
                    issues.append(f'近期跑步配速波动较小且偏慢（平均{avg_pace:.1f} min/km），建议加入强度训练')

        severity = 'medium' if len(issues) >= 1 else 'low'

        return {
            'has_issue': len(issues) > 0,
            'issues': issues,
            'severity': severity
        }

    def _detect_performance_plateau(self, all_df: pd.DataFrame) -> Dict:
        issues = []
        if len(all_df) < 12:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        running = all_df[all_df['sport_type'] == 'running'].copy()
        running = running.sort_values('date_parsed')

        if len(running) < 10:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        mid = len(running) // 2
        first_half = running.iloc[:mid]
        second_half = running.iloc[mid:]

        metrics_to_check = [
            ('avg_pace_min_km', '配速'),
            ('distance_km', '单次跑量'),
        ]
        if 'training_load' in running.columns:
            metrics_to_check.append(('training_load', '训练负荷'))

        all_plateaued = True
        any_data = False
        for metric, label in metrics_to_check:
            if metric in running.columns:
                first_vals = first_half[metric].dropna()
                second_vals = second_half[metric].dropna()
                if len(first_vals) >= 3 and len(second_vals) >= 3:
                    any_data = True
                    first_avg = first_vals.mean()
                    second_avg = second_vals.mean()
                    if first_avg and first_avg > 0:
                        change_pct = abs((second_avg - first_avg) / first_avg * 100)
                        if change_pct > 5:
                            all_plateaued = False

        if all_plateaued and any_data and len(running) >= 10:
            issues.append(f'近{len(running)}次跑步各项指标变化均小于5%，可能进入平台期，建议调整训练计划')

        if 'avg_pace_min_km' in running.columns:
            paces = running['avg_pace_min_km'].dropna().values
            if len(paces) >= 8:
                recent_3 = np.mean(paces[-3:])
                prior_5 = np.mean(paces[-8:-3])
                if prior_5 > 0 and abs(recent_3 - prior_5) / prior_5 < 0.03:
                    issues.append('近3次跑步配速与前5次相比几乎无变化，进步停滞')

        severity = 'medium' if len(issues) >= 1 else 'low'

        return {
            'has_issue': len(issues) > 0,
            'issues': issues,
            'severity': severity
        }

    def _detect_irregular_heart_rate(self, this_week: pd.DataFrame) -> Dict:
        issues = []
        if 'avg_hr' not in this_week.columns or this_week['avg_hr'].notna().sum() < 3:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        hrs = this_week['avg_hr'].dropna()
        if len(hrs) < 3:
            return {'has_issue': False, 'issues': [], 'severity': 'low'}

        hr_std = hrs.std()
        hr_mean = hrs.mean()

        if hr_mean > 0 and hr_std / hr_mean > 0.25:
            issues.append(f'本周心率波动较大（标准差{hr_std:.1f}，均值{hr_mean:.1f}），可能受疲劳或身体状态影响')

        severity = 'medium' if len(issues) > 0 else 'low'

        return {
            'has_issue': len(issues) > 0,
            'issues': issues,
            'severity': severity
        }

    def _generate_charts(self, this_week: pd.DataFrame, all_df: pd.DataFrame,
                         monthly_trend: Optional[Dict] = None) -> Dict:
        charts = {}

        charts['daily_distance'] = self._chart_daily_distance(this_week)
        charts['activity_type_pie'] = self._chart_activity_type(this_week)
        charts['weekly_trend'] = self._chart_weekly_trend(all_df)
        charts['pace_hr_scatter'] = self._chart_pace_hr_scatter(this_week)
        charts['training_load'] = self._chart_training_load(all_df)
        charts['load_recovery_combo'] = self._chart_load_recovery_combo(all_df)
        if monthly_trend and monthly_trend.get('has_data'):
            charts['monthly_trend'] = self._chart_monthly_trend(monthly_trend)

        return charts

    def _chart_daily_distance(self, this_week: pd.DataFrame) -> go.Figure:
        if this_week.empty:
            fig = go.Figure()
            fig.update_layout(title='本周无训练数据', height=350, template='plotly_white')
            return fig

        this_week = this_week.copy()
        this_week['date_only'] = this_week['date_parsed'].dt.date

        fig = go.Figure()

        for sport in sorted(this_week['sport_type'].unique()):
            sport_df = this_week[this_week['sport_type'] == sport]
            daily = sport_df.groupby('date_only')['distance_km'].sum().reset_index()
            daily.columns = ['date', 'distance_km']

            fig.add_trace(go.Bar(
                x=daily['date'],
                y=daily['distance_km'],
                name=self.sport_names.get(sport, sport),
                marker_color=self.sport_colors.get(sport, '#95a5a6'),
                text=[f'{v:.1f}km' if v > 0 else '' for v in daily['distance_km']],
                textposition='outside'
            ))

        fig.update_layout(
            title='本周每日训练距离（按运动类型）',
            xaxis_title='日期',
            yaxis_title='距离 (km)',
            height=350,
            template='plotly_white',
            barmode='stack',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        return fig

    def _chart_activity_type(self, this_week: pd.DataFrame) -> go.Figure:
        if this_week.empty:
            fig = go.Figure()
            fig.update_layout(title='本周无训练数据', height=350, template='plotly_white')
            return fig

        type_counts = this_week['sport_type'].value_counts()
        labels = [self.sport_names.get(t, t) for t in type_counts.index]
        color_list = [self.sport_colors.get(t, '#95a5a6') for t in type_counts.index]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=type_counts.values,
            marker_colors=color_list,
            textinfo='label+percent',
            hole=0.4
        )])
        fig.update_layout(
            title='本周运动类型分布',
            height=350,
            template='plotly_white'
        )
        return fig

    def _chart_weekly_trend(self, all_df: pd.DataFrame) -> go.Figure:
        running = all_df[all_df['sport_type'] == 'running'].copy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if running.empty:
            fig.update_layout(
                title='暂无跑步数据趋势',
                height=400,
                template='plotly_white'
            )
            return fig

        running['week'] = running['date_parsed'].dt.isocalendar().week
        running['year'] = running['date_parsed'].dt.isocalendar().year
        weekly = running.groupby(['year', 'week']).agg({
            'distance_km': 'sum',
            'duration_min': 'sum',
            'avg_pace_min_km': 'mean'
        }).reset_index()
        weekly['week_label'] = weekly.apply(lambda r: f'{int(r["year"])}-W{int(r["week"])}', axis=1)
        weekly = weekly.tail(8)

        fig.add_trace(go.Bar(
            x=weekly['week_label'],
            y=weekly['distance_km'],
            name='周跑量(km)',
            marker_color='#2ecc71',
            text=[f'{v:.1f}km' for v in weekly['distance_km']],
            textposition='outside'
        ), secondary_y=False)

        if weekly['avg_pace_min_km'].notna().any():
            fig.add_trace(go.Scatter(
                x=weekly['week_label'],
                y=weekly['avg_pace_min_km'],
                name='平均配速(min/km)',
                mode='lines+markers',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=8)
            ), secondary_y=True)

        fig.update_layout(
            title='近8周跑步距离与配速趋势',
            height=400,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        fig.update_yaxes(title_text='跑量(km)', secondary_y=False)
        fig.update_yaxes(title_text='配速(min/km)', secondary_y=True, autorange='reversed')
        return fig

    def _chart_pace_hr_scatter(self, this_week: pd.DataFrame) -> go.Figure:
        running = this_week[this_week['sport_type'] == 'running'].copy()
        running = running.dropna(subset=['avg_pace_min_km', 'avg_hr'])

        if running.empty:
            fig = go.Figure()
            fig.update_layout(
                title='暂无配速-心率数据',
                height=350,
                template='plotly_white'
            )
            return fig

        fig = px.scatter(
            running,
            x='avg_pace_min_km',
            y='avg_hr',
            size='distance_km',
            hover_data=['date_parsed', 'distance_km', 'duration_min'],
            title='配速-心率关系（点大小=距离）',
            color_discrete_sequence=['#9b59b6']
        )
        fig.update_layout(
            height=350,
            template='plotly_white',
            xaxis_title='平均配速 (min/km)',
            yaxis_title='平均心率 (bpm)',
            xaxis=dict(autorange='reversed')
        )
        return fig

    def _chart_training_load(self, all_df: pd.DataFrame) -> go.Figure:
        df_with_load = all_df[all_df['training_load'].notna()].copy() if 'training_load' in all_df.columns else pd.DataFrame()

        if df_with_load.empty:
            fig = go.Figure()
            fig.update_layout(
                title='暂无训练负荷数据',
                height=350,
                template='plotly_white'
            )
            return fig

        df_with_load['date_only'] = df_with_load['date_parsed'].dt.date
        daily = df_with_load.groupby('date_only')['training_load'].sum().reset_index()
        daily.columns = ['date', 'load']
        daily = daily.tail(14)

        daily['rolling_7d'] = daily['load'].rolling(window=7, min_periods=1).mean()
        daily['acwr'] = daily['rolling_7d'] / (daily['load'].rolling(window=14, min_periods=1).mean().replace(0, 1))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily['date'],
            y=daily['load'],
            name='每日训练负荷',
            marker_color='#3498db'
        ))
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['rolling_7d'],
            name='7日滚动平均',
            mode='lines',
            line=dict(color='#e74c3c', width=2)
        ))
        fig.update_layout(
            title='近14天训练负荷趋势',
            height=350,
            template='plotly_white',
            xaxis_title='日期',
            yaxis_title='训练负荷',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        return fig

    def _chart_load_recovery_combo(self, all_df: pd.DataFrame) -> go.Figure:
        df_with_load = all_df[all_df['training_load'].notna()].copy() if 'training_load' in all_df.columns else pd.DataFrame()

        if df_with_load.empty:
            fig = go.Figure()
            fig.update_layout(
                title='训练负荷与恢复组合视图（暂无数据）',
                height=450,
                template='plotly_white'
            )
            return fig

        df_with_load['date_only'] = df_with_load['date_parsed'].dt.date

        daily = df_with_load.groupby('date_only').agg({
            'training_load': 'sum',
            'sleep_hours': 'mean',
            'duration_min': 'sum'
        }).reset_index()
        daily.columns = ['date', 'load', 'sleep', 'duration']
        daily = daily.tail(21)

        daily['rolling_7d_load'] = daily['load'].rolling(window=7, min_periods=1).mean()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('训练负荷趋势', '睡眠与恢复'),
            row_heights=[0.6, 0.4]
        )

        fig.add_trace(go.Bar(
            x=daily['date'],
            y=daily['load'],
            name='每日负荷',
            marker_color='#3498db',
            opacity=0.7
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['rolling_7d_load'],
            name='7日滚动平均负荷',
            mode='lines',
            line=dict(color='#e74c3c', width=2)
        ), row=1, col=1)

        sleep_data = daily[daily['sleep'].notna()]
        if not sleep_data.empty:
            fig.add_trace(go.Scatter(
                x=sleep_data['date'],
                y=sleep_data['sleep'],
                name='睡眠时长(小时)',
                mode='lines+markers',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=6)
            ), row=2, col=1)

            fig.add_hline(
                y=7.5,
                line_dash="dash",
                line_color="#f1c40f",
                annotation_text="推荐睡眠",
                row=2, col=1
            )

        fig.update_layout(
            title='训练负荷与恢复状态组合视图',
            height=500,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        fig.update_yaxes(title_text='训练负荷', row=1, col=1)
        fig.update_yaxes(title_text='睡眠(h)', row=2, col=1, range=[0, 12])

        return fig

    def _chart_monthly_trend(self, monthly_trend: Dict) -> go.Figure:
        weekly_data = monthly_trend.get('weekly_data', [])
        if not weekly_data:
            fig = go.Figure()
            fig.update_layout(
                title='月度训练趋势（暂无数据）',
                height=500,
                template='plotly_white'
            )
            return fig

        week_labels = [w['week_label'] for w in weekly_data]
        load_values = [w['total_load'] for w in weekly_data]
        duration_values = [w['total_duration_h'] for w in weekly_data]
        sleep_values = [w['avg_sleep_h'] if w['avg_sleep_h'] is not None else 0 for w in weekly_data]
        injury_markers = [w['has_injury'] for w in weekly_data]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                '周训练负荷趋势',
                '周训练时长趋势',
                '周平均睡眠与伤痛标记'
            ),
            row_heights=[0.4, 0.3, 0.3]
        )

        load_colors = []
        for i, load in enumerate(load_values):
            if i == 0:
                load_colors.append('#3498db')
            elif load > load_values[i - 1] * 1.2:
                load_colors.append('#e74c3c')
            elif load < load_values[i - 1] * 0.8:
                load_colors.append('#f39c12')
            else:
                load_colors.append('#3498db')

        fig.add_trace(go.Bar(
            x=week_labels,
            y=load_values,
            name='周总负荷',
            marker_color=load_colors,
            text=[f'{v:.0f}' for v in load_values],
            textposition='outside'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=week_labels,
            y=duration_values,
            name='训练时长(小时)',
            mode='lines+markers',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=8),
            text=[f'{v:.1f}h' for v in duration_values],
            textposition='top center'
        ), row=2, col=1)

        sleep_plot_x = []
        sleep_plot_y = []
        sleep_colors = []
        for i, (label, sleep, has_injury) in enumerate(zip(week_labels, sleep_values, injury_markers)):
            if sleep > 0:
                sleep_plot_x.append(label)
                sleep_plot_y.append(sleep)
                if has_injury:
                    sleep_colors.append('#e74c3c')
                elif sleep < 6.5:
                    sleep_colors.append('#f39c12')
                else:
                    sleep_colors.append('#2ecc71')

        if sleep_plot_x:
            fig.add_trace(go.Scatter(
                x=sleep_plot_x,
                y=sleep_plot_y,
                name='平均睡眠(小时)',
                mode='lines+markers',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=10, color=sleep_colors,
                           line=dict(color='#000', width=1)),
                text=[f'{v:.1f}h {"⚠️伤痛" if m else ""}' for v, m in zip(sleep_plot_y, injury_markers)]
            ), row=3, col=1)

            fig.add_hline(
                y=7.5,
                line_dash="dash",
                line_color="#f1c40f",
                annotation_text="推荐睡眠",
                row=3, col=1
            )

        for i, (label, has_injury) in enumerate(zip(week_labels, injury_markers)):
            if has_injury:
                fig.add_annotation(
                    x=label,
                    y=1.05,
                    text="⚠️",
                    showarrow=False,
                    font=dict(size=14),
                    xref="x",
                    yref="paper"
                )

        fig.update_layout(
            title='近8周训练负荷、睡眠与恢复趋势',
            height=700,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        fig.update_yaxes(title_text='训练负荷', row=1, col=1)
        fig.update_yaxes(title_text='时长(h)', row=2, col=1)
        fig.update_yaxes(title_text='睡眠(h)', row=3, col=1, range=[0, 12])

        return fig

    def _generate_recovery_combo(self, this_week: pd.DataFrame, all_df: pd.DataFrame,
                                  recovery_analysis: Optional[Dict], anomalies: Dict) -> Dict:
        all_issues = []
        all_recommendations = []

        if recovery_analysis:
            recovery = recovery_analysis.get('recovery_analysis', {})
            if recovery.get('recovery_details'):
                for detail in recovery['recovery_details']:
                    if '状态良好' not in detail and '恢复充足' not in detail:
                        all_issues.append(detail)

            overtraining = recovery_analysis.get('overtraining_risk', {})
            if overtraining.get('indicators'):
                for ind in overtraining['indicators']:
                    if '各项指标正常' not in ind:
                        all_issues.append(ind)

            rest_days = recovery_analysis.get('rest_day_analysis', {})
            if rest_days.get('needs_rest_urgent'):
                all_issues.append(f'连续跑步{rest_days.get("consecutive_days_running", 0)}天，急需休息')

        for anomaly_key, anomaly_data in anomalies.items():
            if isinstance(anomaly_data, dict) and anomaly_data.get('has_issue'):
                for issue in anomaly_data.get('issues', []):
                    all_issues.append(issue)

        if not all_issues:
            all_recommendations.append('训练与恢复状态良好，继续保持当前节奏。')

        total_load = recovery_analysis.get('recovery_analysis', {}).get('7day_total_load', 0) if recovery_analysis else 0
        recovery_score = recovery_analysis.get('recovery_analysis', {}).get('recovery_score', 100) if recovery_analysis else 100
        recovery_level = recovery_analysis.get('recovery_analysis', {}).get('recovery_level', '未知') if recovery_analysis else '未知'

        if recovery_score < 60 or total_load > 1200:
            all_recommendations.append('建议降低本周训练强度，增加1-2天完全休息或轻活动。')
            all_recommendations.append('保证每晚7-8小时睡眠，促进身体恢复。')

        if any('低强度' in str(issue) for issue in all_issues):
            all_recommendations.append('建议加入间歇训练或节奏跑，提升训练强度。')

        if any('平台期' in str(issue) or '停滞' in str(issue) for issue in all_issues):
            all_recommendations.append('可以尝试调整训练计划：增加长距离、加入速度训练或交叉训练。')

        if any('伤痛' in str(issue) for issue in all_issues):
            all_recommendations.append('如有伤痛，请及时就医或休息，不要带伤训练。')

        if not all_recommendations:
            all_recommendations.append('保持当前训练节奏，注意循序渐进。')

        return {
            'issues': all_issues,
            'recommendations': all_recommendations,
            'recovery_score': recovery_score,
            'recovery_level': recovery_level,
            'total_load_7d': total_load,
            'issue_count': len(all_issues)
        }

    def _analyze_monthly_trend(self, df: pd.DataFrame, weeks: int = 8) -> Dict:
        if df.empty:
            return {'has_data': False, 'weeks': [], 'weekly_data': [], 'trend_analysis': '', 'trend_direction': '', 'trend_text': ''}

        df = df.copy()
        df = df.sort_values('date_parsed')

        today = df['date_parsed'].max().to_pydatetime()
        weekly_data = []

        for i in range(weeks - 1, -1, -1):
            week_end = today - timedelta(days=i * 7)
            week_start = week_end - timedelta(days=6)
            week_label = f'{week_start.strftime("%m/%d")}-{week_end.strftime("%m/%d")}'

            week_df = df[
                (df['date_parsed'] >= week_start) &
                (df['date_parsed'] <= week_end)
            ].copy()

            total_load = week_df['training_load'].sum() if 'training_load' in week_df.columns else 0
            total_duration_min = week_df['duration_min'].sum() if 'duration_min' in week_df.columns else 0
            total_duration_h = total_duration_min / 60
            avg_sleep = None
            if 'sleep_hours' in week_df.columns and week_df['sleep_hours'].notna().any():
                avg_sleep = week_df['sleep_hours'].mean()

            has_injury = False
            if 'is_injured' in week_df.columns:
                has_injury = week_df['is_injured'].sum() > 0
            elif 'injury' in week_df.columns:
                has_injury = week_df['injury'].notna().any()

            weekly_data.append({
                'week_label': week_label,
                'week_start': week_start,
                'week_end': week_end,
                'total_load': round(total_load, 1),
                'total_duration_h': round(total_duration_h, 2),
                'total_duration_min': round(total_duration_min, 1),
                'avg_sleep_h': round(avg_sleep, 1) if avg_sleep is not None else None,
                'avg_sleep': round(avg_sleep, 1) if avg_sleep is not None else None,
                'has_injury': has_injury,
                'activity_count': len(week_df)
            })

        load_values = [w['total_load'] for w in weekly_data]
        duration_values = [w['total_duration_h'] for w in weekly_data]
        sleep_values = [w['avg_sleep_h'] for w in weekly_data if w['avg_sleep_h'] is not None]

        trend_analysis_parts = []

        trend_direction = 'stable'

        if len(load_values) >= 2:
            recent_avg = sum(load_values[-2:]) / 2 if load_values[-2:] else 0
            earlier_avg = sum(load_values[:-2]) / max(len(load_values[:-2]), 1) if load_values[:-2] else 0

            if earlier_avg > 0:
                change_pct = (recent_avg - earlier_avg) / earlier_avg * 100
                if change_pct > 20:
                    trend_direction = 'increasing'
                    trend_analysis_parts.append(f'近期训练负荷明显增加（+{change_pct:.0f}%），注意循序渐进避免过度训练')
                elif change_pct < -20:
                    trend_direction = 'decreasing'
                    trend_analysis_parts.append(f'近期训练负荷明显减少（{change_pct:.0f}%），可能处于恢复期或减量期')
                else:
                    trend_direction = 'stable'
                    trend_analysis_parts.append('训练负荷保持稳定，训练持续性良好')
            else:
                trend_analysis_parts.append('训练负荷趋势良好，持续进步中')

        if sleep_values:
            avg_sleep_all = sum(sleep_values) / len(sleep_values)
            if avg_sleep_all < 6.5:
                trend_analysis_parts.append(f'近{len(sleep_values)}周平均睡眠{avg_sleep_all:.1f}小时，睡眠不足可能影响恢复')
            elif avg_sleep_all >= 7.5:
                trend_analysis_parts.append(f'近{len(sleep_values)}周平均睡眠{avg_sleep_all:.1f}小时，睡眠充足恢复良好')

        injury_weeks = [w for w in weekly_data if w['has_injury']]
        if injury_weeks:
            injury_labels = ', '.join([w['week_label'] for w in injury_weeks])
            trend_analysis_parts.append(f'注意：{injury_labels} 有伤痛记录，建议关注恢复情况')

        if not trend_analysis_parts:
            trend_analysis_parts.append('训练趋势稳定，继续保持')

        trend_text = '；'.join(trend_analysis_parts)

        return {
            'has_data': True,
            'weekly_data': weekly_data,
            'weeks': weekly_data,
            'trend_analysis': trend_text,
            'trend_text': trend_text,
            'trend_direction': trend_direction,
            'load_values': load_values,
            'duration_values': duration_values,
            'sleep_values': sleep_values
        }

    def _generate_text_report(self, summary: Dict, anomalies: Dict) -> str:
        this_week = summary.get('this_week', {})
        comparison = summary.get('comparison', {})

        report_lines = []
        report_lines.append('=== 本周训练周报 ===\n')

        by_sport = this_week.get('by_sport', {})
        for sport_key, sport_data in by_sport.items():
            name = sport_data.get('name', sport_key)
            count = sport_data.get('count', 0)
            dist = sport_data.get('distance_km', 0)
            dur = sport_data.get('duration_h', 0)
            if sport_key == 'strength':
                report_lines.append(f'💪 {name} {count}次，时长 {dur:.1f}小时')
            else:
                report_lines.append(f'🏃 {name} {count}次，距离 {dist:.1f}km，时长 {dur:.1f}小时')

        report_lines.append(f'📊 训练负荷总计 {this_week.get("total_training_load", 0):.0f}')
        if this_week.get('avg_sleep_h'):
            report_lines.append(f'😴 平均睡眠 {this_week.get("avg_sleep_h", 0):.1f}小时')

        dist_change = comparison.get('total_distance_km_change_pct', 0)
        if dist_change > 0:
            report_lines.append(f'\n📈 总里程较上周增长 {dist_change:.1f}%')
        elif dist_change < 0:
            report_lines.append(f'\n📉 总里程较上周下降 {abs(dist_change):.1f}%')
        else:
            report_lines.append(f'\n➡️ 总里程与上周基本持平')

        anomaly_issues = []
        for cat, data in anomalies.items():
            if isinstance(data, dict) and data.get('has_issue'):
                anomaly_issues.extend(data.get('issues', []))

        if anomaly_issues:
            report_lines.append('\n⚠️ 异常提醒:')
            for issue in anomaly_issues:
                report_lines.append(f'  • {issue}')
        else:
            report_lines.append('\n✅ 无异常检测，训练状态良好')

        report_lines.append('\n=== 建议 ===')
        if this_week.get('active_days', 0) >= 6:
            report_lines.append('• 本周训练较为密集，建议明日安排休息日')
        if this_week.get('avg_sleep_h') and this_week['avg_sleep_h'] < 7:
            report_lines.append('• 睡眠不足，建议保证每晚7-8小时睡眠')

        running_count = this_week.get('running_count', 0)
        running_dist = this_week.get('running_distance_km', 0)
        if running_count > 0 and running_dist > 0:
            avg_per_run = running_dist / running_count
            if avg_per_run < 5:
                report_lines.append('• 单次跑量偏短，可适当增加长距离训练')

        return '\n'.join(report_lines)

    def _generate_markdown_report(self, summary: Dict, anomalies: Dict,
                                   recovery_analysis: Optional[Dict],
                                   goal_analysis: Optional[Dict],
                                   recovery_combo: Optional[Dict] = None,
                                   monthly_trend: Optional[Dict] = None,
                                   week_period: str = '') -> str:
        this_week = summary.get('this_week', {})
        comparison = summary.get('comparison', {})

        md_lines = []
        md_lines.append('# 📋 训练周报\n')
        md_lines.append(f'**报告周期**: {week_period if week_period else "本周"}\n')
        md_lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        if recovery_combo:
            md_lines.append('## 🎯 核心指标速览\n')
            md_lines.append('> 快速了解本周恢复状态和训练负荷情况\n')
            md_lines.append('| 指标 | 数值 | 说明 |')
            md_lines.append('|------|------|------|')
            md_lines.append(f'| 恢复评分 | {recovery_combo.get("recovery_score", 0)}/100 | {recovery_combo.get("recovery_level", "未知")} |')
            md_lines.append(f'| 7天总负荷 | {recovery_combo.get("total_load_7d", 0):.0f} | 含所有运动类型 |')
            md_lines.append(f'| 发现问题 | {recovery_combo.get("issue_count", 0)} 项 | 需要关注 |')
            md_lines.append('')

        md_lines.append('## 📊 数据概览\n')
        md_lines.append('> 本周与上周训练数据对比\n')
        md_lines.append('| 指标 | 本周 | 上周 | 变化 |')
        md_lines.append('|------|------|------|------|')

        last_week = summary.get('last_week', {})
        metrics = [
            ('总活动次数', 'total_activities', '次'),
            ('总训练时长', 'total_duration_h', '小时'),
            ('总训练距离', 'total_distance_km', 'km'),
            ('总训练负荷', 'total_training_load', ''),
            ('跑步次数', 'running_count', '次'),
            ('跑步距离', 'running_distance_km', 'km'),
        ]

        for label, key, unit in metrics:
            curr = this_week.get(key, 0)
            prev = last_week.get(key, 0)
            change = comparison.get(f'{key}_change_pct', 0)
            change_str = f'+{change:.1f}%' if change >= 0 else f'{change:.1f}%'
            if isinstance(curr, float):
                md_lines.append(f'| {label} | {curr:.1f} {unit} | {prev:.1f} {unit} | {change_str} |')
            else:
                md_lines.append(f'| {label} | {curr} {unit} | {prev} {unit} | {change_str} |')

        md_lines.append('\n## 🏃‍♂️ 运动类型明细\n')
        md_lines.append('> 按运动类型分类的训练统计\n')
        by_sport = this_week.get('by_sport', {})
        if by_sport:
            md_lines.append('| 运动类型 | 次数 | 距离 | 时长 | 负荷 |')
            md_lines.append('|----------|------|------|------|------|')
            for sport_key, sport_data in by_sport.items():
                name = sport_data.get('name', sport_key)
                count = sport_data.get('count', 0)
                dist = sport_data.get('distance_km', 0)
                dur = sport_data.get('duration_h', 0)
                load = sport_data.get('load', 0)
                md_lines.append(f'| {name} | {count} | {dist:.1f}km | {dur:.1f}h | {load:.0f} |')
        else:
            md_lines.append('暂无运动数据')

        if recovery_combo:
            issues = recovery_combo.get('issues', [])
            if issues:
                md_lines.append('\n## ⚠️ 问题汇总\n')
                md_lines.append('> 本周发现的需要关注的问题\n')
                for i, issue in enumerate(issues, 1):
                    md_lines.append(f'{i}. {issue}')

            md_lines.append('\n## 💪 恢复状态详情\n')
            md_lines.append('> 恢复评分、过度训练风险、连续运动天数、睡眠情况\n')
            if recovery_analysis:
                recovery = recovery_analysis.get('recovery_analysis', {})
                score = recovery.get('recovery_score', 0)
                level = recovery.get('recovery_level', '')
                md_lines.append(f'- **恢复评分**: {score}/100 ({level})')
                details = recovery.get('recovery_details', [])
                for d in details:
                    md_lines.append(f'  - {d}')

                overtraining = recovery_analysis.get('overtraining_risk', {})
                risk_level = overtraining.get('risk_level', '')
                md_lines.append(f'- **过度训练风险**: {risk_level}')
                indicators = overtraining.get('indicators', [])
                for ind in indicators:
                    if '各项指标正常' not in ind:
                        md_lines.append(f'  - {ind}')

                rest_days = recovery_analysis.get('rest_day_analysis', {})
                consec_running = rest_days.get('consecutive_days_running', 0)
                consec_any = rest_days.get('consecutive_days_any', 0)
                md_lines.append(f'- **连续运动天数**: 跑步 {consec_running} 天 / 全部运动 {consec_any} 天')

                sleep_stats = recovery_analysis.get('recovery_analysis', {})
                avg_sleep = sleep_stats.get('avg_sleep_h', None)
                if avg_sleep is not None:
                    md_lines.append(f'- **平均睡眠**: {avg_sleep:.1f} 小时/晚')

            anomaly_issues = []
            for cat, data in anomalies.items():
                if isinstance(data, dict) and data.get('has_issue'):
                    anomaly_issues.extend(data.get('issues', []))

            if anomaly_issues:
                md_lines.append('\n### 🚨 异常检测')
                md_lines.append('> 训练过量、低强度、平台期、心率波动等异常检测\n')
                for issue in anomaly_issues:
                    md_lines.append(f'- {issue}')

            recs = recovery_combo.get('recommendations', [])
            if recs:
                md_lines.append('\n## 💡 综合建议\n')
                md_lines.append('> 基于本周数据给出的训练和恢复建议\n')
                for i, rec in enumerate(recs, 1):
                    md_lines.append(f'{i}. {rec}')

        if monthly_trend and monthly_trend.get('has_data', False):
            trend_weeks = len(monthly_trend.get('weekly_data', []))
            md_lines.append(f'\n## 📈 月度趋势（近{trend_weeks}周）\n')
            md_lines.append('> 训练负荷、时长、睡眠和伤痛的长期变化趋势\n')
            weeks = monthly_trend.get('weekly_data', [])
            if weeks:
                md_lines.append('| 周次 | 总负荷 | 训练时长 | 平均睡眠 | 伤痛标记 |')
                md_lines.append('|------|--------|----------|----------|----------|')
                for wk in weeks:
                    week_label = wk.get('week_label', '')
                    load = wk.get('total_load', 0)
                    dur = wk.get('total_duration_h', 0)
                    sleep = wk.get('avg_sleep_h', '-')
                    injury = '⚠️ 有' if wk.get('has_injury', False) else '无'
                    sleep_str = f'{sleep:.1f}h' if isinstance(sleep, (int, float)) else sleep
                    md_lines.append(f'| {week_label} | {load:.0f} | {dur:.1f}h | {sleep_str} | {injury} |')

            trend_analysis = monthly_trend.get('trend_analysis', '')
            if trend_analysis:
                md_lines.append(f'\n**趋势分析**: {trend_analysis}')

            trend_direction = monthly_trend.get('trend_direction', '')
            if trend_direction:
                direction_desc = {
                    'increasing': '📈 训练负荷呈上升趋势，注意循序渐进',
                    'decreasing': '📉 训练负荷呈下降趋势，可能处于恢复期',
                    'stable': '➡️ 训练负荷保持稳定，持续性良好'
                }
                md_lines.append(f'\n**趋势判断**: {direction_desc.get(trend_direction, trend_direction)}')

            injury_weeks_count = sum(1 for w in weeks if w.get('has_injury'))
            if injury_weeks_count > 0:
                md_lines.append(f'\n**伤痛情况**: 近{trend_weeks}周中有 {injury_weeks_count} 周存在伤痛记录，建议关注恢复情况')

        if goal_analysis and goal_analysis.get('has_running_data', True):
            md_lines.append('\n## 🎯 目标进度\n')
            md_lines.append('> 跑量目标完成情况（仅跑步计入）\n')
            weekly = goal_analysis.get('weekly_goal', {})
            monthly = goal_analysis.get('monthly_goal', {})
            yearly = goal_analysis.get('yearly_goal', {})
            if weekly:
                md_lines.append(f'- **周跑量目标**: {weekly.get("distance_so_far_km", 0):.1f} / {weekly.get("goal_km", 0):.1f} km ({weekly.get("progress_pct", 0):.1f}%)')
            if monthly:
                md_lines.append(f'- **月跑量目标**: {monthly.get("distance_so_far_km", 0):.1f} / {monthly.get("goal_km", 0):.1f} km ({monthly.get("progress_pct", 0):.1f}%)')
            if yearly:
                md_lines.append(f'- **年跑量目标**: {yearly.get("distance_so_far_km", 0):.1f} / {yearly.get("goal_km", 0):.1f} km ({yearly.get("progress_pct", 0):.1f}%)')

        md_lines.append('\n---')
        md_lines.append(f'*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')

        return '\n'.join(md_lines)

    def _prepare_export_data(self, all_df: pd.DataFrame, this_week: pd.DataFrame,
                             summary: Dict, recovery_combo: Optional[Dict] = None,
                             anomalies: Optional[Dict] = None,
                             monthly_trend: Optional[Dict] = None,
                             week_period: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
        if all_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        export_cols = [
            'date', 'sport_type', 'distance_km', 'duration_min',
            'avg_pace_min_km', 'elevation_gain_m',
            'avg_hr', 'max_hr', 'calories',
            'sleep_hours', 'injury', 'notes',
            'training_load', 'source_file'
        ]

        available_cols = [c for c in export_cols if c in all_df.columns]
        df_export = all_df[available_cols].copy()

        if 'sport_type' in df_export.columns:
            df_export['sport_type'] = df_export['sport_type'].map(
                lambda x: self.sport_names.get(x, x)
            )

        summary_rows = []
        this_week_stats = summary.get('this_week', {})
        last_week_stats = summary.get('last_week', {})
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary_rows.append({'section': '=== 周报基本信息 ===', 'item': '', 'value': '', 'notes': ''})
        summary_rows.append({'section': '报告周期', 'item': week_period if week_period else '本周', 'value': '', 'notes': '数据截止日期'})
        summary_rows.append({'section': '生成时间', 'item': generated_at, 'value': '', 'notes': '报告生成时间'})

        if recovery_combo:
            summary_rows.append({'section': '=== 核心指标速览 ===', 'item': '', 'value': '', 'notes': ''})
            summary_rows.append({'section': '恢复评分', 'item': f"{recovery_combo.get('recovery_score', 0)}/100", 'value': '', 'notes': recovery_combo.get('recovery_level', '未知')})
            summary_rows.append({'section': '7天总负荷', 'item': f"{recovery_combo.get('total_load_7d', 0):.0f}", 'value': '', 'notes': '含所有运动类型'})
            summary_rows.append({'section': '发现问题数', 'item': f"{recovery_combo.get('issue_count', 0)}", 'value': '', 'notes': '需要关注的问题数量'})

        summary_rows.append({'section': '=== 本周数据概览 ===', 'item': '', 'value': '', 'notes': ''})
        summary_rows.append({'section': '说明', 'item': '本周与上周训练数据对比', 'value': '', 'notes': ''})
        summary_rows.append({'section': '总活动次数', 'item': f"{this_week_stats.get('total_activities', 0)}", 'value': f"{last_week_stats.get('total_activities', 0)}", 'notes': '上周值'})
        summary_rows.append({'section': '总训练时长(小时)', 'item': f"{this_week_stats.get('total_duration_h', 0):.1f}", 'value': f"{last_week_stats.get('total_duration_h', 0):.1f}", 'notes': '上周值'})
        summary_rows.append({'section': '总训练距离(km)', 'item': f"{this_week_stats.get('total_distance_km', 0):.1f}", 'value': f"{last_week_stats.get('total_distance_km', 0):.1f}", 'notes': '上周值'})
        summary_rows.append({'section': '总训练负荷', 'item': f"{this_week_stats.get('total_training_load', 0):.0f}", 'value': f"{last_week_stats.get('total_training_load', 0):.0f}", 'notes': '上周值'})

        by_sport = this_week_stats.get('by_sport', {})
        if by_sport:
            summary_rows.append({'section': '=== 运动类型明细 ===', 'item': '', 'value': '', 'notes': ''})
            summary_rows.append({'section': '说明', 'item': '按运动类型分类的训练统计', 'value': '', 'notes': ''})
            for sport_key, sport_data in by_sport.items():
                name = sport_data.get('name', sport_key)
                count = sport_data.get('count', 0)
                dist = sport_data.get('distance_km', 0)
                dur = sport_data.get('duration_h', 0)
                load = sport_data.get('load', 0)
                summary_rows.append({
                    'section': name,
                    'item': f"{count}次",
                    'value': f"{dist:.1f}km / {dur:.1f}h",
                    'notes': f"负荷{load:.0f}"
                })

        if recovery_combo:
            issues = recovery_combo.get('issues', [])
            if issues:
                summary_rows.append({'section': '=== 问题汇总 ===', 'item': '', 'value': '', 'notes': ''})
                summary_rows.append({'section': '说明', 'item': '本周发现的需要关注的问题', 'value': '', 'notes': ''})
                for i, issue in enumerate(issues, 1):
                    summary_rows.append({'section': f'问题{i}', 'item': issue, 'value': '', 'notes': ''})

            recs = recovery_combo.get('recommendations', [])
            if recs:
                summary_rows.append({'section': '=== 综合建议 ===', 'item': '', 'value': '', 'notes': ''})
                summary_rows.append({'section': '说明', 'item': '基于本周数据给出的训练和恢复建议', 'value': '', 'notes': ''})
                for i, rec in enumerate(recs, 1):
                    summary_rows.append({'section': f'建议{i}', 'item': rec, 'value': '', 'notes': ''})

        if anomalies:
            anomaly_issues = []
            for cat, data in anomalies.items():
                if isinstance(data, dict) and data.get('has_issue'):
                    anomaly_issues.extend(data.get('issues', []))
            if anomaly_issues:
                summary_rows.append({'section': '=== 异常检测 ===', 'item': '', 'value': '', 'notes': ''})
                summary_rows.append({'section': '说明', 'item': '训练过量、低强度、平台期、心率波动等异常检测', 'value': '', 'notes': ''})
                for i, issue in enumerate(anomaly_issues, 1):
                    summary_rows.append({'section': f'异常{i}', 'item': issue, 'value': '', 'notes': ''})

        if monthly_trend and monthly_trend.get('has_data'):
            weekly_data = monthly_trend.get('weekly_data', [])
            trend_weeks = len(weekly_data)
            if weekly_data:
                summary_rows.append({'section': f'=== 近{trend_weeks}周趋势 ===', 'item': '', 'value': '', 'notes': ''})
                summary_rows.append({'section': '说明', 'item': '训练负荷、时长、睡眠和伤痛的长期变化趋势', 'value': '', 'notes': ''})
                for wk in weekly_data:
                    injury_flag = ' ⚠️伤痛' if wk.get('has_injury', False) else ''
                    sleep_str = f"{wk.get('avg_sleep_h', 0):.1f}h" if wk.get('avg_sleep_h') else '无睡眠'
                    summary_rows.append({
                        'section': wk.get('week_label', ''),
                        'item': f"负荷{wk.get('total_load', 0):.0f}",
                        'value': f"{wk.get('total_duration_h', 0):.1f}h",
                        'notes': f"{sleep_str}{injury_flag}"
                    })
            trend_analysis = monthly_trend.get('trend_analysis', '')
            if trend_analysis:
                summary_rows.append({'section': '趋势分析', 'item': trend_analysis, 'value': '', 'notes': ''})

            trend_direction = monthly_trend.get('trend_direction', '')
            if trend_direction:
                direction_desc = {
                    'increasing': '训练负荷呈上升趋势，注意循序渐进',
                    'decreasing': '训练负荷呈下降趋势，可能处于恢复期',
                    'stable': '训练负荷保持稳定，持续性良好'
                }
                summary_rows.append({'section': '趋势判断', 'item': direction_desc.get(trend_direction, trend_direction), 'value': '', 'notes': ''})

            injury_weeks_count = sum(1 for w in weekly_data if w.get('has_injury'))
            if injury_weeks_count > 0:
                summary_rows.append({'section': '伤痛情况', 'item': f'近{trend_weeks}周中有 {injury_weeks_count} 周存在伤痛记录，建议关注恢复情况', 'value': '', 'notes': ''})

        summary_rows.append({'section': f'=== 报告结束 ===', 'item': '', 'value': '', 'notes': ''})
        summary_rows.append({'section': '报告生成时间', 'item': generated_at, 'value': '', 'notes': ''})

        df_summary = pd.DataFrame(summary_rows)

        return df_export, df_summary

    def export_csv(self, export_data: pd.DataFrame) -> str:
        if export_data.empty:
            return ''
        return export_data.to_csv(index=False, encoding='utf-8-sig')

    def export_markdown(self, markdown_report: str) -> str:
        return markdown_report
