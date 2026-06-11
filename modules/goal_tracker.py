import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from calendar import monthrange


class GoalTracker:
    def __init__(self,
                 monthly_distance_goal_km: float = 150.0,
                 weekly_distance_goal_km: float = 40.0,
                 yearly_distance_goal_km: float = 1800.0):
        self.monthly_distance_goal_km = monthly_distance_goal_km
        self.weekly_distance_goal_km = weekly_distance_goal_km
        self.yearly_distance_goal_km = yearly_distance_goal_km

    def analyze(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}

        df = df.copy()
        running_df = df[df['sport_type'] == 'running'].copy()

        weekly_goal = self._track_weekly_goal(running_df)
        monthly_goal = self._track_monthly_goal(running_df)
        yearly_goal = self._track_yearly_goal(running_df)
        streak_info = self._calculate_streaks(running_df)
        trend = self._analyze_progress_trend(running_df)

        return {
            'weekly_goal': weekly_goal,
            'monthly_goal': monthly_goal,
            'yearly_goal': yearly_goal,
            'streaks': streak_info,
            'trend': trend,
            'goals_config': {
                'weekly_distance_km': self.weekly_distance_goal_km,
                'monthly_distance_km': self.monthly_distance_goal_km,
                'yearly_distance_km': self.yearly_distance_goal_km
            }
        }

    def _track_weekly_goal(self, running_df: pd.DataFrame) -> Dict:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)

        week_df = running_df[
            (running_df['date_parsed'] >= week_start) &
            (running_df['date_parsed'] <= week_end)
        ].copy()

        distance_so_far = week_df['distance_km'].sum()
        runs_count = len(week_df)
        duration_min = week_df['duration_min'].sum()
        avg_pace = week_df['avg_pace_min_km'].mean() if week_df['avg_pace_min_km'].notna().any() else None

        days_passed = today.weekday() + 1
        days_total = 7
        progress_pct = (distance_so_far / self.weekly_distance_goal_km * 100) if self.weekly_distance_goal_km > 0 else 0
        expected_pct = (days_passed / days_total * 100)
        pace_diff = progress_pct - expected_pct

        remaining = max(0, self.weekly_distance_goal_km - distance_so_far)
        days_left = days_total - days_passed
        avg_needed = remaining / days_left if days_left > 0 else 0

        return {
            'period': f'{week_start.strftime("%Y-%m-%d")} ~ {week_end.strftime("%Y-%m-%d")}',
            'distance_so_far_km': round(distance_so_far, 2),
            'goal_km': self.weekly_distance_goal_km,
            'remaining_km': round(remaining, 2),
            'progress_pct': round(progress_pct, 1),
            'expected_pct': round(expected_pct, 1),
            'pace_status': '超前' if pace_diff > 5 else ('落后' if pace_diff < -5 else '按计划'),
            'pace_diff_pct': round(pace_diff, 1),
            'runs_count': runs_count,
            'total_duration_h': round(duration_min / 60, 2),
            'avg_pace': self._format_pace(avg_pace) if avg_pace else '--:--',
            'days_passed': days_passed,
            'days_left': days_left,
            'avg_needed_per_day_km': round(avg_needed, 2),
            'on_track': pace_diff >= -10
        }

    def _track_monthly_goal(self, running_df: pd.DataFrame) -> Dict:
        today = datetime.now()
        month_start = today.replace(day=1)
        next_month = month_start.replace(month=month_start.month + 1) if month_start.month < 12 else month_start.replace(year=month_start.year + 1, month=1)
        month_end = next_month - timedelta(days=1)
        _, days_in_month = monthrange(today.year, today.month)

        month_df = running_df[
            (running_df['date_parsed'] >= month_start) &
            (running_df['date_parsed'] <= month_end)
        ].copy()

        distance_so_far = month_df['distance_km'].sum()
        runs_count = len(month_df)
        duration_min = month_df['duration_min'].sum()
        avg_pace = month_df['avg_pace_min_km'].mean() if month_df['avg_pace_min_km'].notna().any() else None
        total_elevation = month_df['elevation_gain_m'].sum() if 'elevation_gain_m' in month_df.columns else 0
        avg_load = month_df['training_load'].mean() if 'training_load' in month_df.columns and month_df['training_load'].notna().any() else None

        days_passed = today.day
        progress_pct = (distance_so_far / self.monthly_distance_goal_km * 100) if self.monthly_distance_goal_km > 0 else 0
        expected_pct = (days_passed / days_in_month * 100)
        pace_diff = progress_pct - expected_pct

        remaining = max(0, self.monthly_distance_goal_km - distance_so_far)
        days_left = days_in_month - days_passed
        avg_needed = remaining / days_left if days_left > 0 else 0

        last_3_months = self._get_past_months_data(running_df, today, 3)

        return {
            'period': f'{today.year}年{today.month}月',
            'distance_so_far_km': round(distance_so_far, 2),
            'goal_km': self.monthly_distance_goal_km,
            'remaining_km': round(remaining, 2),
            'progress_pct': round(progress_pct, 1),
            'expected_pct': round(expected_pct, 1),
            'pace_status': '超前' if pace_diff > 5 else ('落后' if pace_diff < -5 else '按计划'),
            'pace_diff_pct': round(pace_diff, 1),
            'runs_count': runs_count,
            'total_duration_h': round(duration_min / 60, 2),
            'avg_pace': self._format_pace(avg_pace) if avg_pace else '--:--',
            'total_elevation_m': round(total_elevation, 1),
            'avg_training_load': round(avg_load, 1) if avg_load else None,
            'days_passed': days_passed,
            'days_in_month': days_in_month,
            'days_left': days_left,
            'avg_needed_per_day_km': round(avg_needed, 2),
            'avg_per_run_km': round(distance_so_far / runs_count, 2) if runs_count > 0 else 0,
            'on_track': pace_diff >= -10,
            'past_3_months': last_3_months
        }

    def _get_past_months_data(self, running_df: pd.DataFrame, today: datetime, months_count: int) -> List[Dict]:
        result = []
        for i in range(1, months_count + 1):
            month_date = today - timedelta(days=i * 30)
            month_start = month_date.replace(day=1)
            if month_start.month == 12:
                next_month = month_start.replace(year=month_start.year + 1, month=1)
            else:
                next_month = month_start.replace(month=month_start.month + 1)
            month_end = next_month - timedelta(days=1)

            month_df = running_df[
                (running_df['date_parsed'] >= month_start) &
                (running_df['date_parsed'] <= month_end)
            ]
            result.append({
                'month': f'{month_start.year}-{month_start.month:02d}',
                'distance_km': round(month_df['distance_km'].sum(), 2),
                'runs': len(month_df)
            })
        return list(reversed(result))

    def _track_yearly_goal(self, running_df: pd.DataFrame) -> Dict:
        today = datetime.now()
        year_start = today.replace(month=1, day=1)
        year_end = today.replace(month=12, day=31)
        days_in_year = 366 if (today.year % 4 == 0 and today.year % 100 != 0) or today.year % 400 == 0 else 365

        year_df = running_df[
            (running_df['date_parsed'] >= year_start) &
            (running_df['date_parsed'] <= year_end)
        ].copy()

        distance_so_far = year_df['distance_km'].sum()
        runs_count = len(year_df)
        duration_min = year_df['duration_min'].sum()

        day_of_year = today.timetuple().tm_yday
        progress_pct = (distance_so_far / self.yearly_distance_goal_km * 100) if self.yearly_distance_goal_km > 0 else 0
        expected_pct = (day_of_year / days_in_year * 100)
        pace_diff = progress_pct - expected_pct

        remaining = max(0, self.yearly_distance_goal_km - distance_so_far)
        days_left = days_in_year - day_of_year
        avg_needed = remaining / days_left if days_left > 0 else 0

        return {
            'period': f'{today.year}年',
            'distance_so_far_km': round(distance_so_far, 2),
            'goal_km': self.yearly_distance_goal_km,
            'remaining_km': round(remaining, 2),
            'progress_pct': round(progress_pct, 1),
            'expected_pct': round(expected_pct, 1),
            'pace_status': '超前' if pace_diff > 5 else ('落后' if pace_diff < -5 else '按计划'),
            'pace_diff_pct': round(pace_diff, 1),
            'runs_count': runs_count,
            'total_duration_h': round(duration_min / 60, 2),
            'days_passed': day_of_year,
            'days_in_year': days_in_year,
            'days_left': days_left,
            'avg_needed_per_day_km': round(avg_needed, 2),
            'on_track': pace_diff >= -10
        }

    def _calculate_streaks(self, running_df: pd.DataFrame) -> Dict:
        if running_df.empty:
            return {
                'current_streak_days': 0,
                'longest_streak_days': 0,
                'current_week_streak': 0
            }

        running_dates = sorted(set(running_df['date_parsed'].dt.date))
        today = datetime.now().date()

        current_streak = 0
        check_date = today
        while check_date in running_dates:
            current_streak += 1
            check_date -= timedelta(days=1)

        longest_streak = 0
        temp_streak = 1
        for i in range(1, len(running_dates)):
            diff = (running_dates[i] - running_dates[i-1]).days
            if diff == 1:
                temp_streak += 1
            else:
                longest_streak = max(longest_streak, temp_streak)
                temp_streak = 1
        longest_streak = max(longest_streak, temp_streak)

        this_week_start = today - timedelta(days=today.weekday())
        week_run_days = sum(1 for d in running_dates if d >= this_week_start and d <= today)

        return {
            'current_streak_days': current_streak,
            'longest_streak_days': longest_streak,
            'current_week_run_days': week_run_days
        }

    def _analyze_progress_trend(self, running_df: pd.DataFrame) -> Dict:
        if len(running_df) < 10:
            return {
                'status': '数据不足',
                'weekly_avg_recent': 0,
                'weekly_avg_prior': 0,
                'change_pct': 0,
                'pace_trend': '稳定',
                'pace_change_pct': 0
            }

        df_sorted = running_df.sort_values('date_parsed')

        total_weeks = len(df_sorted)
        mid_point = total_weeks // 2

        recent = df_sorted.tail(mid_point)
        prior = df_sorted.head(mid_point)

        recent_avg_dist = recent['distance_km'].mean()
        prior_avg_dist = prior['distance_km'].mean()

        if prior_avg_dist > 0:
            change_pct = ((recent_avg_dist - prior_avg_dist) / prior_avg_dist) * 100
        else:
            change_pct = 0

        recent_paces = recent['avg_pace_min_km'].dropna()
        prior_paces = prior['avg_pace_min_km'].dropna()

        pace_trend = '稳定'
        pace_change_pct = 0
        if len(recent_paces) >= 3 and len(prior_paces) >= 3:
            recent_avg_pace = recent_paces.mean()
            prior_avg_pace = prior_paces.mean()
            if prior_avg_pace > 0:
                pace_change_pct = ((prior_avg_pace - recent_avg_pace) / prior_avg_pace) * 100
                if pace_change_pct > 3:
                    pace_trend = '进步'
                elif pace_change_pct < -3:
                    pace_trend = '退步'

        if change_pct > 10:
            status = '里程上升趋势'
        elif change_pct < -10:
            status = '里程下降趋势'
        else:
            status = '里程稳定'

        return {
            'status': status,
            'weekly_avg_recent_km': round(recent_avg_dist, 2),
            'weekly_avg_prior_km': round(prior_avg_dist, 2),
            'distance_change_pct': round(change_pct, 1),
            'pace_trend': pace_trend,
            'pace_change_pct': round(pace_change_pct, 1)
        }

    def _format_pace(self, pace_min_km: Optional[float]) -> str:
        if pace_min_km is None or pd.isna(pace_min_km) or pace_min_km <= 0:
            return '--:--'
        total_seconds = int(round(pace_min_km * 60))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f'{minutes:02d}:{seconds:02d}'
