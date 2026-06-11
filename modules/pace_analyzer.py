import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


class PaceAnalyzer:
    def __init__(self, resting_hr: float = 60.0, max_hr: Optional[float] = None):
        self.resting_hr = resting_hr
        self.max_hr = max_hr if max_hr else self._estimate_max_hr(age=35)

    def _estimate_max_hr(self, age: int) -> float:
        return 220.0 - float(age)

    def calculate_training_load(self, duration_min: float, avg_hr: Optional[float],
                                max_hr_activity: Optional[float]) -> float:
        if not avg_hr or not duration_min:
            return 0.0
        hrr = self.max_hr - self.resting_hr
        if hrr <= 0:
            return 0.0
        hr_ratio = (avg_hr - self.resting_hr) / hrr
        hr_ratio = max(0.0, min(1.5, hr_ratio))
        load = duration_min * hr_ratio
        return round(load, 1)

    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        if df.empty:
            return df, {}

        df = df.copy()

        df['training_load'] = df.apply(
            lambda row: self.calculate_training_load(
                row.get('duration_min', 0),
                row.get('avg_hr'),
                row.get('max_hr')
            ),
            axis=1
        )

        df['pace_min_km'] = df.apply(
            lambda row: self._format_pace(row.get('avg_pace_min_km')),
            axis=1
        )

        df['speed_kmh'] = df.apply(
            lambda row: round(60.0 / row['avg_pace_min_km'], 2)
            if row.get('avg_pace_min_km') and row['avg_pace_min_km'] > 0 else None,
            axis=1
        )

        df['effort_score'] = df.apply(
            lambda row: self._calc_effort_score(row),
            axis=1
        )

        running_df = df[df['sport_type'] == 'running'].copy()
        summary = self._generate_summary(running_df, df)

        pace_distribution = self._analyze_pace_distribution(running_df)
        elevation_stats = self._analyze_elevation(running_df)
        hr_stats = self._analyze_heart_rate(df)

        analysis = {
            'summary': summary,
            'pace_distribution': pace_distribution,
            'elevation_stats': elevation_stats,
            'heart_rate_stats': hr_stats
        }

        return df, analysis

    def _format_pace(self, pace_min_km: Optional[float]) -> str:
        if pace_min_km is None or pd.isna(pace_min_km) or pace_min_km <= 0:
            return '--:--'
        total_seconds = int(round(pace_min_km * 60))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f'{minutes:02d}:{seconds:02d}'

    def _calc_effort_score(self, row: pd.Series) -> float:
        score = 0.0
        pace = row.get('avg_pace_min_km')
        duration = row.get('duration_min', 0)
        elevation = row.get('elevation_gain_m', 0) or 0
        avg_hr = row.get('avg_hr')

        if pace and pace > 0:
            if pace <= 5:
                score += 40
            elif pace <= 6:
                score += 30
            elif pace <= 7:
                score += 20
            else:
                score += 10

        if duration >= 60:
            score += 30
        elif duration >= 30:
            score += 20
        elif duration >= 15:
            score += 10

        if elevation >= 300:
            score += 20
        elif elevation >= 100:
            score += 10

        if avg_hr and self.max_hr > 0:
            hr_pct = avg_hr / self.max_hr
            if hr_pct >= 0.9:
                score += 10
            elif hr_pct >= 0.8:
                score += 5

        return round(score, 1)

    def _generate_summary(self, running_df: pd.DataFrame, all_df: pd.DataFrame) -> Dict:
        if running_df.empty:
            return {}

        total_runs = len(running_df)
        total_distance = running_df['distance_km'].sum()
        total_duration = running_df['duration_min'].sum()
        avg_pace = running_df['avg_pace_min_km'].mean() if 'avg_pace_min_km' in running_df.columns and running_df['avg_pace_min_km'].notna().any() else None
        best_pace = running_df['avg_pace_min_km'].min() if 'avg_pace_min_km' in running_df.columns and running_df['avg_pace_min_km'].notna().any() else None
        avg_hr = running_df['avg_hr'].mean() if 'avg_hr' in running_df.columns and running_df['avg_hr'].notna().any() else None
        max_hr = running_df['max_hr'].max() if 'max_hr' in running_df.columns and running_df['max_hr'].notna().any() else None
        total_elevation = running_df['elevation_gain_m'].sum() if 'elevation_gain_m' in running_df.columns else 0
        total_load = running_df['training_load'].sum() if 'training_load' in running_df.columns else 0

        return {
            'total_runs': total_runs,
            'total_distance_km': round(total_distance, 2),
            'total_duration_h': round(total_duration / 60, 2),
            'avg_pace': self._format_pace(avg_pace) if avg_pace else '--:--',
            'best_pace': self._format_pace(best_pace) if best_pace else '--:--',
            'avg_hr': round(avg_hr, 1) if avg_hr else None,
            'max_hr_recorded': round(max_hr, 1) if max_hr else None,
            'total_elevation_m': round(total_elevation, 1),
            'total_training_load': round(total_load, 1),
            'avg_distance_per_run_km': round(total_distance / total_runs, 2) if total_runs > 0 else 0
        }

    def _analyze_pace_distribution(self, running_df: pd.DataFrame) -> Dict:
        if running_df.empty or 'avg_pace_min_km' not in running_df.columns:
            return {}

        pace_data = running_df['avg_pace_min_km'].dropna()
        if pace_data.empty:
            return {}

        bins = [0, 5, 5.5, 6, 6.5, 7, 8, 10, float('inf')]
        labels = ['<5:00', '5:00-5:30', '5:30-6:00', '6:00-6:30', '6:30-7:00', '7:00-8:00', '8:00-10:00', '>10:00']
        pace_cats = pd.cut(pace_data, bins=bins, labels=labels, right=False)
        counts = pace_cats.value_counts().sort_index()

        return {
            'distribution': counts.to_dict(),
            'median_pace': self._format_pace(pace_data.median()),
            'pace_std': round(pace_data.std(), 2) if len(pace_data) > 1 else 0,
            'pace_q25': self._format_pace(pace_data.quantile(0.25)),
            'pace_q75': self._format_pace(pace_data.quantile(0.75))
        }

    def _analyze_elevation(self, running_df: pd.DataFrame) -> Dict:
        if running_df.empty:
            return {}

        elev_data = running_df['elevation_gain_m'].dropna() if 'elevation_gain_m' in running_df.columns else pd.Series()
        if elev_data.empty:
            return {}

        flat = len(elev_data[elev_data < 30])
        rolling = len(elev_data[(elev_data >= 30) & (elev_data < 150)])
        hilly = len(elev_data[(elev_data >= 150) & (elev_data < 500)])
        mountainous = len(elev_data[elev_data >= 500])

        return {
            'avg_elevation_m': round(elev_data.mean(), 1),
            'max_elevation_m': round(elev_data.max(), 1),
            'terrain_distribution': {
                '平坦 (<30m)': flat,
                '起伏 (30-150m)': rolling,
                '丘陵 (150-500m)': hilly,
                '山地 (>=500m)': mountainous
            }
        }

    def _analyze_heart_rate(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}

        avg_hr_data = df['avg_hr'].dropna() if 'avg_hr' in df.columns else pd.Series()
        max_hr_data = df['max_hr'].dropna() if 'max_hr' in df.columns else pd.Series()

        result = {}
        if not avg_hr_data.empty:
            result['avg_hr_overall'] = round(avg_hr_data.mean(), 1)
            result['avg_hr_min'] = round(avg_hr_data.min(), 1)
            result['avg_hr_max'] = round(avg_hr_data.max(), 1)
        if not max_hr_data.empty:
            result['max_hr_overall'] = round(max_hr_data.max(), 1)

        return result
