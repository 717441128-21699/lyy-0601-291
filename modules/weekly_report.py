import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class WeeklyReport:
    def __init__(self):
        pass

    def generate(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}

        df = df.copy()
        df = df.sort_values('date_parsed')

        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        last_week_start = week_start - timedelta(days=7)

        this_week_df = df[df['date_parsed'] >= week_start].copy()
        last_week_df = df[(df['date_parsed'] >= last_week_start) & (df['date_parsed'] < week_start)].copy()

        week_summary = self._calculate_week_summary(this_week_df, last_week_df)
        anomalies = self._detect_anomalies(df, this_week_df)
        charts = self._generate_charts(this_week_df, df)
        text_report = self._generate_text_report(week_summary, anomalies)

        return {
            'week_period': f'{week_start.strftime("%Y-%m-%d")} ~ {today.strftime("%Y-%m-%d")}',
            'summary': week_summary,
            'anomalies': anomalies,
            'charts': charts,
            'text_report': text_report
        }

    def _calculate_week_summary(self, this_week: pd.DataFrame, last_week: pd.DataFrame) -> Dict:
        def calc_stats(df: pd.DataFrame) -> Dict:
            running = df[df['sport_type'] == 'running']
            cycling = df[df['sport_type'] == 'cycling']
            strength = df[df['sport_type'] == 'strength']

            return {
                'total_activities': len(df),
                'running_count': len(running),
                'cycling_count': len(cycling),
                'strength_count': len(strength),
                'running_distance_km': round(running['distance_km'].sum(), 2),
                'running_duration_h': round(running['duration_min'].sum() / 60, 2),
                'cycling_distance_km': round(cycling['distance_km'].sum(), 2),
                'cycling_duration_h': round(cycling['duration_min'].sum() / 60, 2),
                'strength_duration_h': round(strength['duration_min'].sum() / 60, 2),
                'total_duration_h': round(df['duration_min'].sum() / 60, 2),
                'total_elevation_m': round(running['elevation_gain_m'].sum(), 1) if 'elevation_gain_m' in running.columns else 0,
                'total_training_load': round(df['training_load'].sum(), 1) if 'training_load' in df.columns else 0,
                'avg_hr': round(df['avg_hr'].mean(), 1) if 'avg_hr' in df.columns and df['avg_hr'].notna().any() else None,
                'avg_pace': running['avg_pace_min_km'].mean() if 'avg_pace_min_km' in running.columns and running['avg_pace_min_km'].notna().any() else None,
                'active_days': df['date_parsed'].dt.date.nunique(),
                'avg_sleep_h': round(df['sleep_hours'].mean(), 1) if 'sleep_hours' in df.columns and df['sleep_hours'].notna().any() else None,
                'injury_count': int(df['is_injured'].sum()) if 'is_injured' in df.columns else 0
            }

        this_week_stats = calc_stats(this_week)
        last_week_stats = calc_stats(last_week)

        def calc_change(curr, prev):
            if prev and prev != 0 and isinstance(curr, (int, float)):
                return round((curr - prev) / prev * 100, 1)
            return 0

        comparison = {}
        for key in ['running_distance_km', 'running_duration_h', 'total_duration_h',
                     'total_training_load', 'running_count']:
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
            return {'has_issue': False, 'issues': []}

        running = this_week[this_week['sport_type'] == 'running']
        total_distance = running['distance_km'].sum()
        total_duration = running['duration_min'].sum()
        total_load = running['training_load'].sum() if 'training_load' in running.columns else 0

        all_running = all_df[all_df['sport_type'] == 'running']
        if len(all_running) >= 10:
            weekly_dists = all_running.groupby(all_running['date_parsed'].dt.isocalendar().week)['distance_km'].sum()
            if len(weekly_dists) >= 3:
                avg_weekly = weekly_dists.mean()
                std_weekly = weekly_dists.std()
                if total_distance > avg_weekly + 2 * std_weekly and total_distance > 60:
                    issues.append(f'本周跑量{total_distance:.1f}km显著高于历史平均（{avg_weekly:.1f}km），增加量超过2个标准差')

        if total_distance > 80:
            issues.append(f'本周跑量{total_distance:.1f}km超过80km，注意恢复')

        if total_duration > 600:
            issues.append(f'本周跑步时长{total_duration/60:.1f}小时超过10小时')

        if total_load > 1500:
            issues.append(f'本周训练负荷{total_load:.0f}过高，建议降低强度')

        running_days = running['date_parsed'].dt.date.nunique()
        if running_days >= 6:
            issues.append(f'本周跑步{running_days}天，建议安排至少1个休息日')

        return {
            'has_issue': len(issues) > 0,
            'issues': issues,
            'total_distance_km': round(total_distance, 2),
            'total_duration_h': round(total_duration / 60, 2),
            'total_load': round(total_load, 1)
        }

    def _detect_low_intensity_stagnation(self, all_df: pd.DataFrame) -> Dict:
        issues = []
        if len(all_df) < 10:
            return {'has_issue': False, 'issues': []}

        running = all_df[all_df['sport_type'] == 'running'].copy()
        running = running.sort_values('date_parsed')

        if len(running) < 8:
            return {'has_issue': False, 'issues': []}

        recent = running.tail(8)

        if 'avg_hr' in recent.columns and recent['avg_hr'].notna().sum() >= 5:
            max_hr = running['max_hr'].max() if 'max_hr' in running.columns else 180
            if max_hr and max_hr > 0:
                hr_pcts = [hr / max_hr for hr in recent['avg_hr'].dropna() if hr]
                if hr_pcts and all(p < 0.65 for p in hr_pcts):
                    issues.append(f'近{len(hr_pcts)}次训练平均心率均低于最大心率的65%，长期低强度训练可能限制进步')

        if 'avg_pace_min_km' in recent.columns and recent['avg_pace_min_km'].notna().sum() >= 5:
            paces = recent['avg_pace_min_km'].dropna().values
            if len(paces) >= 5:
                pace_std = np.std(paces)
                avg_pace = np.mean(paces)
                if avg_pace > 0 and pace_std / avg_pace < 0.08 and avg_pace > 7:
                    issues.append(f'近期配速波动较小且偏慢（平均{avg_pace:.1f} min/km），建议加入强度训练')

        return {
            'has_issue': len(issues) > 0,
            'issues': issues
        }

    def _detect_performance_plateau(self, all_df: pd.DataFrame) -> Dict:
        issues = []
        if len(all_df) < 15:
            return {'has_issue': False, 'issues': []}

        running = all_df[all_df['sport_type'] == 'running'].copy()
        running = running.sort_values('date_parsed')

        if len(running) < 12:
            return {'has_issue': False, 'issues': []}

        mid = len(running) // 2
        first_half = running.iloc[:mid]
        second_half = running.iloc[mid:]

        metrics_to_check = [
            ('avg_pace_min_km', '配速'),
            ('distance_km', '单次跑量'),
            ('training_load', '训练负荷') if 'training_load' in running.columns else None
        ]
        metrics_to_check = [m for m in metrics_to_check if m]

        all_plateaued = True
        for metric, label in metrics_to_check:
            if metric in running.columns:
                first_avg = first_half[metric].dropna().mean()
                second_avg = second_half[metric].dropna().mean()
                if first_avg and first_avg > 0:
                    change_pct = abs((second_avg - first_avg) / first_avg * 100)
                    if change_pct > 5:
                        all_plateaued = False
                        break

        if all_plateaued and len(running) >= 12:
            issues.append(f'近{len(running)}次训练各项指标变化均小于5%，可能进入平台期，建议调整训练计划')

        paces = running['avg_pace_min_km'].dropna().values
        if len(paces) >= 8:
            recent_3 = np.mean(paces[-3:])
            prior_5 = np.mean(paces[-8:-3])
            if prior_5 > 0 and recent_3 >= prior_5 * 0.98 and recent_3 <= prior_5 * 1.05:
                issues.append('近3次跑步配速与前5次相比几乎无变化，进步停滞')

        return {
            'has_issue': len(issues) > 0,
            'issues': issues
        }

    def _detect_irregular_heart_rate(self, this_week: pd.DataFrame) -> Dict:
        issues = []
        if 'avg_hr' not in this_week.columns or this_week['avg_hr'].notna().sum() < 3:
            return {'has_issue': False, 'issues': []}

        hrs = this_week['avg_hr'].dropna()
        if len(hrs) < 3:
            return {'has_issue': False, 'issues': []}

        hr_std = hrs.std()
        hr_mean = hrs.mean()

        if hr_mean > 0 and hr_std / hr_mean > 0.25:
            issues.append(f'本周心率波动较大（标准差{hr_std:.1f}，均值{hr_mean:.1f}），可能受疲劳或身体状态影响')

        return {
            'has_issue': len(issues) > 0,
            'issues': issues
        }

    def _generate_charts(self, this_week: pd.DataFrame, all_df: pd.DataFrame) -> Dict:
        charts = {}

        charts['daily_distance'] = self._chart_daily_distance(this_week)
        charts['activity_type_pie'] = self._chart_activity_type(this_week)
        charts['weekly_trend'] = self._chart_weekly_trend(all_df)
        charts['pace_hr_scatter'] = self._chart_pace_hr_scatter(this_week)
        charts['training_load'] = self._chart_training_load(all_df)

        return charts

    def _chart_daily_distance(self, this_week: pd.DataFrame) -> go.Figure:
        running = this_week[this_week['sport_type'] == 'running'].copy()
        if running.empty:
            return go.Figure()

        running['date_only'] = running['date_parsed'].dt.date
        daily = running.groupby('date_only')['distance_km'].sum().reset_index()
        daily.columns = ['date', 'distance_km']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily['date'],
            y=daily['distance_km'],
            name='跑步距离(km)',
            marker_color='#2ecc71',
            text=[f'{v:.1f}km' for v in daily['distance_km']],
            textposition='outside'
        ))
        fig.update_layout(
            title='本周每日跑步距离',
            xaxis_title='日期',
            yaxis_title='距离 (km)',
            height=350,
            template='plotly_white'
        )
        return fig

    def _chart_activity_type(self, this_week: pd.DataFrame) -> go.Figure:
        if this_week.empty:
            return go.Figure()

        type_counts = this_week['sport_type'].value_counts()
        labels_map = {'running': '跑步', 'cycling': '骑行', 'strength': '力量训练', 'other': '其他'}
        labels = [labels_map.get(t, t) for t in type_counts.index]
        colors = {'running': '#2ecc71', 'cycling': '#3498db', 'strength': '#e67e22', 'other': '#95a5a6'}
        color_list = [colors.get(t, '#95a5a6') for t in type_counts.index]

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
        if running.empty:
            return go.Figure()

        running['week'] = running['date_parsed'].dt.isocalendar().week
        running['year'] = running['date_parsed'].dt.isocalendar().year
        weekly = running.groupby(['year', 'week']).agg({
            'distance_km': 'sum',
            'duration_min': 'sum',
            'avg_pace_min_km': 'mean'
        }).reset_index()
        weekly['week_label'] = weekly.apply(lambda r: f'{int(r["year"])}-W{int(r["week"])}', axis=1)
        weekly = weekly.tail(8)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=weekly['week_label'],
            y=weekly['distance_km'],
            name='周跑量(km)',
            marker_color='#3498db',
            text=[f'{v:.1f}km' for v in weekly['distance_km']],
            textposition='outside'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=weekly['week_label'],
            y=weekly['avg_pace_min_km'],
            name='平均配速(min/km)',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=8)
        ), secondary_y=True)

        fig.update_layout(
            title='近8周跑量与配速趋势',
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
            return go.Figure()

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
        running = all_df[all_df['sport_type'] == 'running'].copy()
        if running.empty or 'training_load' not in running.columns:
            return go.Figure()

        running['date_only'] = running['date_parsed'].dt.date
        daily = running.groupby('date_only')['training_load'].sum().reset_index()
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

    def _generate_text_report(self, summary: Dict, anomalies: Dict) -> str:
        this_week = summary.get('this_week', {})
        comparison = summary.get('comparison', {})

        report_lines = []
        report_lines.append('=== 本周训练周报 ===\n')

        report_lines.append(f'🏃 跑步 {this_week.get("running_count", 0)}次，距离 {this_week.get("running_distance_km", 0):.1f}km，时长 {this_week.get("running_duration_h", 0):.1f}小时')
        report_lines.append(f'🚴 骑行 {this_week.get("cycling_count", 0)}次，距离 {this_week.get("cycling_distance_km", 0):.1f}km')
        report_lines.append(f'💪 力量训练 {this_week.get("strength_count", 0)}次，时长 {this_week.get("strength_duration_h", 0):.1f}小时')
        report_lines.append(f'📊 训练负荷总计 {this_week.get("total_training_load", 0):.0f}')
        if this_week.get('avg_sleep_h'):
            report_lines.append(f'😴 平均睡眠 {this_week.get("avg_sleep_h", 0):.1f}小时')

        dist_change = comparison.get('running_distance_km_change_pct', 0)
        if dist_change > 0:
            report_lines.append(f'\n📈 跑量较上周增长 {dist_change:.1f}%')
        elif dist_change < 0:
            report_lines.append(f'\n📉 跑量较上周下降 {abs(dist_change):.1f}%')
        else:
            report_lines.append(f'\n➡️ 跑量与上周基本持平')

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
        if this_week.get('running_distance_km', 0) > 0 and this_week.get('running_count', 0) > 0:
            avg_per_run = this_week['running_distance_km'] / this_week['running_count']
            if avg_per_run < 5:
                report_lines.append('• 单次跑量偏短，可适当增加长距离训练')

        return '\n'.join(report_lines)
