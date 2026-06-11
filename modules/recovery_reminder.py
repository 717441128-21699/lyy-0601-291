import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class RecoveryReminder:
    def __init__(self, recommended_sleep_hours: float = 7.5,
                 consecutive_runs_before_rest: int = 3,
                 weekly_rest_days: int = 1):
        self.recommended_sleep_hours = recommended_sleep_hours
        self.consecutive_runs_before_rest = consecutive_runs_before_rest
        self.weekly_rest_days = weekly_rest_days

    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        if df.empty:
            return df, {
                'recovery_analysis': {},
                'rest_day_analysis': {},
                'overtraining_risk': {},
                'reminders': [],
                'injury_count': 0,
                'sleep_issues_count': 0,
                'has_data': False
            }

        df = df.copy()
        df = df.sort_values('date_parsed').reset_index(drop=True)

        injury_flags = self._flag_injuries(df)
        df['is_injured'] = injury_flags

        sleep_flags = self._analyze_sleep(df)
        df['sleep_quality'] = sleep_flags['quality']
        df['sleep_flag'] = sleep_flags['flag']

        recovery_analysis = self._analyze_recovery_needs(df)
        rest_day_analysis = self._check_rest_days(df)
        overtraining_risk = self._assess_overtraining_risk(df)

        reminders = self._generate_reminders(df, recovery_analysis, rest_day_analysis, overtraining_risk)

        analysis = {
            'recovery_analysis': recovery_analysis,
            'rest_day_analysis': rest_day_analysis,
            'overtraining_risk': overtraining_risk,
            'reminders': reminders,
            'injury_count': int(df['is_injured'].sum()) if 'is_injured' in df.columns else 0,
            'sleep_issues_count': int((df['sleep_flag'] == True).sum()) if 'sleep_flag' in df.columns else 0,
            'has_data': True
        }

        return df, analysis

    def _flag_injuries(self, df: pd.DataFrame) -> List[bool]:
        flags = []
        for _, row in df.iterrows():
            injury = str(row.get('injury', '')).lower().strip()
            notes = str(row.get('notes', '')).lower().strip()
            pain_keywords = [
                '伤痛', '疼痛', '受伤', '痛', 'injury', 'pain', 'hurt', 'sore',
                '膝盖', '跟腱', '脚踝', '胫骨', '小腿', '大腿', '背部',
                'knee', 'ankle', 'achilles', 'shin', 'calf', 'thigh', 'back',
                '劳损', '拉伤', '扭伤', 'strain', 'sprain'
            ]
            is_injured = any(k in injury for k in pain_keywords) or any(k in notes for k in pain_keywords)
            flags.append(is_injured)
        return flags

    def _analyze_sleep(self, df: pd.DataFrame) -> Dict:
        qualities = []
        flags = []
        for _, row in df.iterrows():
            sleep = row.get('sleep_hours')
            if pd.isna(sleep) or sleep is None or sleep == 0:
                qualities.append('无数据')
                flags.append(False)
            elif sleep < 5:
                qualities.append('严重不足')
                flags.append(True)
            elif sleep < 6:
                qualities.append('不足')
                flags.append(True)
            elif sleep < self.recommended_sleep_hours:
                qualities.append('一般')
                flags.append(False)
            elif sleep <= 9:
                qualities.append('良好')
                flags.append(False)
            else:
                qualities.append('过多')
                flags.append(False)
        return {'quality': qualities, 'flag': flags}

    def _analyze_recovery_needs(self, df: pd.DataFrame) -> Dict:
        today = datetime.now()
        last_7_days = today - timedelta(days=7)

        recent_df = df[df['date_parsed'] >= last_7_days].copy()
        recent_running = recent_df[recent_df['sport_type'] == 'running']
        recent_all = recent_df[recent_df['sport_type'].isin(['running', 'cycling', 'strength', 'walking', 'swimming'])]

        total_load_all = recent_all['training_load'].sum() if 'training_load' in recent_all.columns else 0
        total_load_running = recent_running['training_load'].sum() if 'training_load' in recent_running.columns else 0

        total_distance_running = recent_running['distance_km'].sum()
        total_duration_running = recent_running['duration_min'].sum()
        total_duration_all = recent_all['duration_min'].sum()

        days_with_activity = recent_all['date_parsed'].dt.date.nunique()
        days_with_running = recent_running['date_parsed'].dt.date.nunique()

        avg_sleep = None
        if 'sleep_hours' in recent_df.columns and recent_df['sleep_hours'].notna().any():
            avg_sleep = recent_df['sleep_hours'].dropna().mean()

        recovery_score = self._calculate_recovery_score(total_load_all, days_with_activity, avg_sleep)

        return {
            '7day_total_load': round(total_load_all, 1),
            '7day_running_load': round(total_load_running, 1),
            '7day_total_distance_km': round(total_distance_running, 2),
            '7day_total_duration_h': round(total_duration_all / 60, 2),
            '7day_running_duration_h': round(total_duration_running / 60, 2),
            '7day_days_with_activity': int(days_with_activity),
            '7day_days_with_running': int(days_with_running),
            '7day_avg_sleep_h': round(avg_sleep, 1) if avg_sleep else None,
            'recovery_score': recovery_score['score'],
            'recovery_level': recovery_score['level'],
            'recovery_details': recovery_score['details']
        }

    def _calculate_recovery_score(self, load: float, days_active: int, avg_sleep: Optional[float]) -> Dict:
        score = 100
        details = []

        if load > 1500:
            score -= 30
            details.append('近7天训练负荷很高')
        elif load > 1000:
            score -= 20
            details.append('近7天训练负荷较高')
        elif load > 600:
            score -= 10
            details.append('近7天训练负荷中等')

        if days_active >= 6:
            score -= 20
            details.append('近7天训练天数过多，缺少休息日')
        elif days_active >= 5:
            score -= 10
            details.append('近7天训练较频繁')

        if avg_sleep is not None:
            if avg_sleep < 5.5:
                score -= 30
                details.append(f'平均睡眠仅{avg_sleep:.1f}小时，严重不足')
            elif avg_sleep < 6.5:
                score -= 20
                details.append(f'平均睡眠{avg_sleep:.1f}小时，不足')
            elif avg_sleep < 7.5:
                score -= 10
                details.append(f'平均睡眠{avg_sleep:.1f}小时，略低于推荐量')
            elif avg_sleep >= 8:
                score += 10
                details.append(f'平均睡眠{avg_sleep:.1f}小时，恢复充足')

        score = max(0, min(100, score))

        if score >= 80:
            level = '恢复良好'
        elif score >= 60:
            level = '恢复一般'
        elif score >= 40:
            level = '需要休息'
        else:
            level = '过度疲劳风险'

        if not details:
            details.append('状态良好')

        return {'score': score, 'level': level, 'details': details}

    def _check_rest_days(self, df: pd.DataFrame) -> Dict:
        today = datetime.now()
        last_7_days = today - timedelta(days=7)

        recent_df = df[df['date_parsed'] >= last_7_days].copy()
        active_dates = set(recent_df[recent_df['sport_type'].isin(
            ['running', 'cycling', 'strength', 'walking', 'swimming']
        )]['date_parsed'].dt.date)

        week_start = today - timedelta(days=today.weekday())
        all_dates = set()
        for i in range(7):
            all_dates.add((week_start + timedelta(days=i)).date())

        rest_dates = sorted(all_dates - active_dates)

        streak_info = self._get_current_streak(df, today)

        return {
            'rest_dates_this_week': [d.isoformat() for d in rest_dates],
            'rest_days_count_this_week': len(rest_dates),
            'consecutive_days_running': streak_info['running_streak'],
            'consecutive_days_any': streak_info['any_streak'],
            'needs_rest_urgent': streak_info['running_streak'] >= self.consecutive_runs_before_rest,
            'needs_rest_any': streak_info['any_streak'] >= self.consecutive_runs_before_rest + 1
        }

    def _get_current_streak(self, df: pd.DataFrame, today: datetime) -> Dict:
        running_dates = set(df[df['sport_type'] == 'running']['date_parsed'].dt.date)
        any_dates = set(df[df['sport_type'].isin(
            ['running', 'cycling', 'strength', 'walking', 'swimming']
        )]['date_parsed'].dt.date)

        running_streak = 0
        any_streak = 0
        check_date = today.date()

        while check_date in running_dates:
            running_streak += 1
            check_date -= timedelta(days=1)

        check_date = today.date()
        while check_date in any_dates:
            any_streak += 1
            check_date -= timedelta(days=1)

        return {
            'running_streak': running_streak,
            'any_streak': any_streak
        }

    def _assess_overtraining_risk(self, df: pd.DataFrame) -> Dict:
        if len(df) < 3:
            return {
                'risk_level': '数据不足',
                'risk_score': 0,
                'indicators': ['数据量不足，建议积累更多训练记录后评估'],
                'has_data': False
            }

        df_sorted = df.sort_values('date_parsed')
        risk_score = 0
        indicators = []

        if 'training_load' in df.columns and df['training_load'].notna().any():
            recent_loads = df_sorted.tail(14)['training_load'].values
            valid_loads = [l for l in recent_loads if l and not pd.isna(l)]
            if len(valid_loads) >= 4:
                n = min(7, len(valid_loads) // 2)
                if n > 0:
                    recent_avg = np.mean(valid_loads[-n:])
                    prior_avg = np.mean(valid_loads[:-n]) if len(valid_loads[:-n]) > 0 else recent_avg
                    if prior_avg > 0 and recent_avg / prior_avg > 1.5:
                        risk_score += 30
                        indicators.append(f'近期训练负荷环比增长{((recent_avg/prior_avg)-1)*100:.0f}%')

        last_date = df_sorted['date_parsed'].max()
        today = datetime.now()
        since_last = (today - last_date).days if pd.notna(last_date) else 999

        if since_last <= 2:
            running_df = df_sorted[df_sorted['sport_type'] == 'running']
            if len(running_df) >= 5:
                paces = running_df['avg_pace_min_km'].dropna().values
                if len(paces) >= 5:
                    n_recent = min(3, len(paces) // 3)
                    if n_recent > 0 and len(paces) > n_recent:
                        recent_pace = np.mean(paces[-n_recent:])
                        earlier_pace = np.mean(paces[:-n_recent])
                        if earlier_pace > 0 and recent_pace > earlier_pace * 1.08:
                            risk_score += 25
                            indicators.append('近期配速明显下降，可能存在疲劳')

        if 'sleep_hours' in df.columns:
            recent_sleep = df_sorted.tail(7)['sleep_hours'].dropna()
            if len(recent_sleep) >= 3 and recent_sleep.mean() < 6.5:
                risk_score += 25
                indicators.append(f'近7天平均睡眠仅{recent_sleep.mean():.1f}小时')

        injuries = df_sorted.tail(14)['is_injured'].sum() if 'is_injured' in df_sorted.columns else 0
        if injuries > 0:
            risk_score += 20
            indicators.append(f'近期存在{int(injuries)}条伤痛记录')

        if risk_score >= 60:
            level = '高风险'
        elif risk_score >= 35:
            level = '中风险'
        elif risk_score >= 15:
            level = '低风险'
        else:
            level = '无明显风险'

        return {
            'risk_level': level,
            'risk_score': risk_score,
            'indicators': indicators if indicators else ['各项指标正常'],
            'has_data': True
        }

    def _generate_reminders(self, df: pd.DataFrame, recovery: Dict,
                            rest_days: Dict, overtraining: Dict) -> List[Dict]:
        reminders = []

        if rest_days.get('needs_rest_urgent'):
            reminders.append({
                'type': 'warning',
                'priority': 'high',
                'title': '急需休息日',
                'message': f'已连续跑步{rest_days["consecutive_days_running"]}天，建议今天或明天安排完全休息日'
            })

        if recovery.get('recovery_level') in ['需要休息', '过度疲劳风险']:
            reminders.append({
                'type': 'warning',
                'priority': 'high',
                'title': f'恢复状态: {recovery["recovery_level"]}',
                'message': '; '.join(recovery.get('recovery_details', []))
            })

        if overtraining.get('risk_level') in ['中风险', '高风险']:
            reminders.append({
                'type': 'danger',
                'priority': 'high',
                'title': f'过度训练{overtraining["risk_level"]}',
                'message': '; '.join(overtraining.get('indicators', []))
            })

        if rest_days.get('rest_days_count_this_week', 7) < self.weekly_rest_days:
            reminders.append({
                'type': 'info',
                'priority': 'medium',
                'title': '休息日提醒',
                'message': f'本周仅休息{rest_days.get("rest_days_count_this_week", 0)}天，建议至少安排{self.weekly_rest_days}个休息日'
            })

        today = datetime.now().date()
        recent_injuries = df[(df['is_injured'] == True) &
                             (df['date_parsed'].dt.date >= today - timedelta(days=14))]
        if not recent_injuries.empty:
            reminders.append({
                'type': 'warning',
                'priority': 'high',
                'title': '伤痛提醒',
                'message': f'近两周有{len(recent_injuries)}条伤痛记录，建议充分恢复后再训练'
            })

        if not reminders:
            reminders.append({
                'type': 'success',
                'priority': 'low',
                'title': '状态良好',
                'message': '恢复状态良好，继续保持！'
            })

        return reminders
