import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List


class ActivityCleaner:
    def __init__(self):
        self.reasonable_thresholds = {
            'running': {
                'min_pace': 2.5,
                'max_pace': 15.0,
                'max_distance_single': 80.0,
                'max_duration_single': 720.0
            },
            'cycling': {
                'min_pace': 3.0,
                'max_pace': 50.0,
                'max_distance_single': 250.0,
                'max_duration_single': 720.0
            },
            'strength': {
                'min_duration': 5.0,
                'max_duration': 240.0
            }
        }

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, pd.DataFrame()

        df = df.copy()
        issues = []

        df = self._parse_dates(df)
        df, dup_issues = self._remove_duplicates(df)
        issues.extend(dup_issues)

        df, type_issues = self._identify_sport_types(df)
        issues.extend(type_issues)

        df, corr_issues = self._correct_distance_duration(df)
        issues.extend(corr_issues)

        df, clean_issues = self._clean_outliers(df)
        issues.extend(clean_issues)

        df = self._ensure_fields(df)
        df = df.sort_values('date_parsed', ascending=False).reset_index(drop=True)

        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(
            columns=['date', 'sport_type', 'issue_type', 'details'])
        return df, issues_df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        def parse_single_date(date_str):
            if not date_str or pd.isna(date_str) or str(date_str).strip() == '':
                return pd.NaT
            date_str = str(date_str).strip()
            for fmt in [
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d',
                '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M', '%Y/%m/%d',
                '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%d/%m/%Y',
                '%m/%d/%Y %H:%M:%S', '%m/%d/%Y'
            ]:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    pass
            try:
                return pd.to_datetime(date_str, utc=True).tz_localize(None)
            except:
                try:
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                except:
                    return pd.NaT

        df['date_parsed'] = df['date'].apply(parse_single_date)
        df = df.dropna(subset=['date_parsed']).reset_index(drop=True)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()

        df['date_floor'] = df['date_parsed'].dt.floor('h')

        def is_duplicate(group):
            if len(group) <= 1:
                return group.iloc[0:0]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    row1 = group.iloc[i]
                    row2 = group.iloc[j]
                    dist_diff = abs(row1.get('distance_km', 0) - row2.get('distance_km', 0))
                    dur_diff = abs(row1.get('duration_min', 0) - row2.get('duration_min', 0))
                    if (row1.get('distance_km', 0) > 0 and row2.get('distance_km', 0) > 0 and dist_diff < 0.5 and
                        row1.get('duration_min', 0) > 0 and row2.get('duration_min', 0) > 0 and dur_diff < 5):
                        return group.iloc[[j]]
            return group.iloc[0:0]

        dup_rows = []
        for _, group in df.groupby(['date_floor', 'sport_type']):
            dups = is_duplicate(group)
            if not dups.empty:
                dup_rows.extend(dups.index.tolist())
                for _, row in dups.iterrows():
                    issues.append({
                        'date': row.get('date', ''),
                        'sport_type': row.get('sport_type', ''),
                        'issue_type': '重复记录',
                        'details': f'检测到重复记录，已删除。距离: {row.get("distance_km", 0):.2f}km, 时长: {row.get("duration_min", 0):.1f}min'
                    })

        df = df.drop(index=dup_rows).drop(columns=['date_floor']).reset_index(drop=True)
        return df, issues

    def _identify_sport_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()

        for idx, row in df.iterrows():
            original_type = str(row.get('sport_type', '')).lower().strip()
            distance = row.get('distance_km', 0) or 0
            duration = row.get('duration_min', 0) or 0
            pace = row.get('avg_pace_min_km', None)
            detected_type = None

            strength_keywords = ['strength', '力量', '力量训练', 'weight', 'gym', '体能', 'crossfit']
            cycling_keywords = ['cycling', '骑行', '自行车', 'bike', 'biking']
            running_keywords = ['running', '跑步', 'run', 'jog', '慢跑']

            if any(k in original_type for k in strength_keywords):
                detected_type = 'strength'
            elif any(k in original_type for k in cycling_keywords):
                detected_type = 'cycling'
            elif any(k in original_type for k in running_keywords) or original_type in ['run', '']:
                detected_type = 'running'

            if detected_type is None:
                if pace and 2.5 <= pace <= 15.0 and distance > 0:
                    detected_type = 'running'
                elif pace and 3.0 <= pace <= 50.0 and distance > 5:
                    detected_type = 'cycling'
                elif distance == 0 and duration > 0:
                    detected_type = 'strength'
                else:
                    detected_type = 'other'

            if detected_type != original_type and original_type != '':
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': original_type,
                    'issue_type': '运动类型修正',
                    'details': f'从 "{original_type}" 修正为 "{detected_type}" (基于配速 {pace:.1f} min/km, 距离 {distance:.2f}km)'
                })

            df.at[idx, 'sport_type'] = detected_type

        return df, issues

    def _correct_distance_duration(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()

        for idx, row in df.iterrows():
            sport = row.get('sport_type', 'running')
            distance = row.get('distance_km', 0) or 0
            duration = row.get('duration_min', 0) or 0
            pace = row.get('avg_pace_min_km', None)
            corrected = False
            details = []

            if sport in ['running', 'cycling']:
                if distance <= 0 and pace and duration > 0:
                    distance = duration / pace if pace > 0 else 0
                    df.at[idx, 'distance_km'] = distance
                    corrected = True
                    details.append(f'距离已修正: {distance:.2f}km (基于配速和时长)')

                if duration <= 0 and pace and distance > 0:
                    duration = distance * pace
                    df.at[idx, 'duration_min'] = duration
                    corrected = True
                    details.append(f'时长已修正: {duration:.1f}min (基于配速和距离)')

                if pace and distance > 0 and duration > 0:
                    calc_pace = duration / distance
                    if abs(calc_pace - pace) > 1.0:
                        df.at[idx, 'avg_pace_min_km'] = calc_pace
                        corrected = True
                        details.append(f'配速已修正: {calc_pace:.2f} min/km (原始:{pace:.2f})')

                if sport == 'running':
                    if distance > self.reasonable_thresholds['running']['max_distance_single']:
                        issues.append({
                            'date': row.get('date', ''),
                            'sport_type': sport,
                            'issue_type': '异常数据警告',
                            'details': f'跑步距离 {distance:.2f}km 超出正常范围（阈值: {self.reasonable_thresholds["running"]["max_distance_single"]}km）'
                        })
                    if duration > self.reasonable_thresholds['running']['max_duration_single']:
                        issues.append({
                            'date': row.get('date', ''),
                            'sport_type': sport,
                            'issue_type': '异常数据警告',
                            'details': f'跑步时长 {duration:.1f}min 超出正常范围'
                        })

            elif sport == 'strength':
                if duration <= 0 and distance > 0:
                    df.at[idx, 'distance_km'] = 0.0
                    details.append('力量训练距离已归零')
                    corrected = True

            if corrected and details:
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': sport,
                    'issue_type': '数据修正',
                    'details': '; '.join(details)
                })

        return df, issues

    def _clean_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()
        to_drop = []

        for idx, row in df.iterrows():
            sport = row.get('sport_type', 'running')
            distance = row.get('distance_km', 0) or 0
            duration = row.get('duration_min', 0) or 0

            if sport in ['running', 'cycling']:
                if distance < 0.05 and duration < 1:
                    to_drop.append(idx)
                    issues.append({
                        'date': row.get('date', ''),
                        'sport_type': sport,
                        'issue_type': '无效记录',
                        'details': f'距离过短 ({distance:.3f}km) 且时长短 ({duration:.1f}min)，已删除'
                    })
            elif sport == 'strength':
                if duration < 2:
                    to_drop.append(idx)
                    issues.append({
                        'date': row.get('date', ''),
                        'sport_type': sport,
                        'issue_type': '无效记录',
                        'details': f'力量训练时长短 ({duration:.1f}min)，已删除'
                    })

        df = df.drop(index=to_drop).reset_index(drop=True)
        return df, issues

    def _ensure_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        required_fields = [
            'date', 'date_parsed', 'sport_type', 'distance_km', 'duration_min',
            'elevation_gain_m', 'avg_hr', 'max_hr', 'avg_pace_min_km',
            'calories', 'sleep_hours', 'injury', 'notes', 'source_file', 'track_points'
        ]
        for field in required_fields:
            if field not in df.columns:
                if field == 'date_parsed':
                    continue
                df[field] = None if field != 'track_points' else [[] for _ in range(len(df))]

        numeric_fields = ['distance_km', 'duration_min', 'elevation_gain_m',
                          'avg_hr', 'max_hr', 'avg_pace_min_km', 'calories', 'sleep_hours']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')

        return df
