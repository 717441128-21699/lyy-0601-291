import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional


class ActivityCleaner:
    def __init__(self):
        self.reasonable_thresholds = {
            'running': {
                'min_pace': 2.5,
                'max_pace': 15.0,
                'min_distance_km': 0.05,
                'max_distance_single': 100.0,
                'max_duration_single': 720.0,
                'typical_speed_kmh': (8, 20)
            },
            'cycling': {
                'min_pace': 1.5,
                'max_pace': 10.0,
                'min_distance_km': 0.5,
                'max_distance_single': 300.0,
                'max_duration_single': 720.0,
                'typical_speed_kmh': (15, 45)
            },
            'strength': {
                'min_duration': 2.0,
                'max_duration': 240.0
            },
            'walking': {
                'min_pace': 8.0,
                'max_pace': 25.0,
                'min_distance_km': 0.1,
                'max_distance_single': 30.0,
                'typical_speed_kmh': (3, 7)
            }
        }

        self.sport_keywords = {
            'running': [
                'run', 'running', '跑步', '慢跑', 'jog', 'jogging',
                'race', '马拉松', 'marathon', '越野跑', 'trail',
                'interval', '间歇', 'tempo', '节奏跑', '长距离', 'long run',
                '晨跑', '夜跑', '跑', 'easy run', 'speed workout', 'lSD'
            ],
            'cycling': [
                'cycle', 'cycling', 'bike', 'biking', '骑行', '自行车',
                'road bike', 'mtb', '山地车', '公路车', 'spin', '动感单车',
                '骑车', '单车', 'ride', 'riding'
            ],
            'strength': [
                'strength', 'weight', 'weights', 'resistance', '力量',
                'gym', '健身', '体能', 'crossfit', 'hiit', '力量训练',
                '举重', '哑铃', '杠铃', '深蹲', '硬拉', '卧推', '俯卧撑', 'plank',
                '健身房', '器械', '核心', '臀腿', '胸背', '上肢', '下肢'
            ],
            'walking': [
                'walk', 'walking', '步行', '散步', 'hike', 'hiking', '徒步',
                '远足', '健走', '遛弯', '逛街'
            ],
            'swimming': [
                'swim', 'swimming', '游泳', '泳池', '自由泳', '蛙泳',
                '仰泳', '蝶泳', '泳池游泳', 'open water'
            ]
        }

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, pd.DataFrame(columns=['date', 'sport_type', 'issue_type', 'severity', 'details'])

        df = df.copy()
        issues = []

        df = self._parse_dates(df)
        df, dup_issues = self._remove_duplicates(df)
        issues.extend(dup_issues)

        df, type_issues = self._identify_sport_types(df)
        issues.extend(type_issues)

        df, corr_issues = self._correct_distance_duration(df)
        issues.extend(corr_issues)

        df, field_issues = self._handle_missing_fields(df)
        issues.extend(field_issues)

        df, clean_issues = self._clean_outliers(df)
        issues.extend(clean_issues)

        df = self._ensure_fields(df)
        df = df.sort_values('date_parsed', ascending=False).reset_index(drop=True)

        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(
            columns=['date', 'sport_type', 'issue_type', 'severity', 'details'])
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

        def find_duplicates(group):
            if len(group) <= 1:
                return []
            dup_indices = []
            checked = set()
            for i in range(len(group)):
                if i in checked:
                    continue
                for j in range(i + 1, len(group)):
                    if j in checked:
                        continue
                    row1 = group.iloc[i]
                    row2 = group.iloc[j]
                    dist1 = row1.get('distance_km', 0) or 0
                    dist2 = row2.get('distance_km', 0) or 0
                    dur1 = row1.get('duration_min', 0) or 0
                    dur2 = row2.get('duration_min', 0) or 0

                    dist_diff = abs(dist1 - dist2)
                    dur_diff = abs(dur1 - dur2)

                    is_dup = False
                    if dist1 > 0 and dist2 > 0 and dur1 > 0 and dur2 > 0:
                        if dist_diff < 0.5 and dur_diff < 5:
                            is_dup = True
                    elif dist1 > 0 and dist2 > 0 and dist_diff < 0.3:
                        is_dup = True
                    elif dur1 > 0 and dur2 > 0 and dur_diff < 2:
                        is_dup = True

                    if is_dup and j not in dup_indices:
                        dup_indices.append(group.index[j])
                        checked.add(j)
            return dup_indices

        dup_rows = []
        for _, group in df.groupby(['date_floor', 'sport_type']):
            dups = find_duplicates(group)
            for idx in dups:
                dup_rows.append(idx)
                row = df.loc[idx]
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': row.get('sport_type', ''),
                    'issue_type': '重复记录',
                    'severity': 'warning',
                    'details': f'检测到重复记录，已删除。距离: {row.get("distance_km", 0):.2f}km, 时长: {row.get("duration_min", 0):.1f}min'
                })

        df = df.drop(index=dup_rows).drop(columns=['date_floor']).reset_index(drop=True)
        return df, issues

    def _identify_sport_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()

        for idx, row in df.iterrows():
            original_type = str(row.get('sport_type', '')).lower().strip()
            notes = str(row.get('notes', '')).lower().strip()
            name = str(row.get('name', '')).lower().strip()
            distance = row.get('distance_km', 0) or 0
            duration = row.get('duration_min', 0) or 0
            pace = row.get('avg_pace_min_km', None)

            if pd.isna(distance):
                distance = 0
            if pd.isna(duration):
                duration = 0
            if pd.isna(pace):
                pace = None

            speed_kmh = None
            if distance > 0 and duration > 0:
                speed_kmh = distance / (duration / 60.0)

            detected_type, confidence, reason = self._classify_sport(
                original_type, notes, name, distance, duration, pace, speed_kmh
            )

            df.at[idx, 'sport_type'] = detected_type
            df.at[idx, 'sport_confidence'] = confidence

            if original_type and original_type != detected_type and original_type not in ['', 'unknown', 'other', '未分类']:
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': original_type,
                    'issue_type': '运动类型修正',
                    'severity': 'info',
                    'details': f'从 "{original_type}" 修正为 "{detected_type}" (置信度: {confidence}%)。原因: {reason}'
                })
            elif not original_type or original_type in ['', 'unknown', 'other', '未分类']:
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': detected_type,
                    'issue_type': '运动类型识别',
                    'severity': 'info',
                    'details': f'自动识别为 "{detected_type}" (置信度: {confidence}%)。原因: {reason}'
                })

        return df, issues

    def _classify_sport(self, type_str: str, notes: str, name: str,
                        distance: float, duration: float,
                        pace: Optional[float], speed_kmh: Optional[float]) -> Tuple[str, int, str]:
        combined_text = f"{type_str} {notes} {name}".lower()

        scores = {sport: 0 for sport in self.sport_keywords.keys()}

        for sport, keywords in self.sport_keywords.items():
            for kw in keywords:
                if kw.lower() in combined_text:
                    scores[sport] += 30

        max_score = max(scores.values()) if scores else 0
        if max_score >= 30:
            top_sport = max(scores, key=scores.get)
            return top_sport, min(100, max_score + 10), f'匹配关键词 "{top_sport}"'

        if speed_kmh and speed_kmh > 0:
            if 8 <= speed_kmh <= 22:
                scores['running'] += 25
            elif 15 <= speed_kmh <= 50:
                scores['cycling'] += 35
            elif 3 <= speed_kmh <= 7:
                scores['walking'] += 25

        if pace and pace > 0:
            if 2.5 <= pace <= 8:
                scores['running'] += 15
            elif 1.5 <= pace <= 5:
                scores['cycling'] += 20
            elif 8 <= pace <= 25:
                scores['walking'] += 15

        if distance == 0 and duration > 5:
            scores['strength'] += 25

        if distance > 0 and duration > 0:
            if distance >= 3 and distance <= 20 and duration >= 15 and duration <= 120:
                scores['running'] += 10
            if distance >= 10 and distance <= 200:
                scores['cycling'] += 5
            if distance <= 3 and duration >= 20:
                scores['strength'] += 10
                scores['walking'] += 5

        if distance > 80:
            scores['cycling'] += 20
            scores['running'] -= 10

        max_score = max(scores.values()) if scores else 0

        if max_score >= 20:
            top_sport = max(scores, key=scores.get)
            reasons = []
            if type_str:
                reasons.append(f'类型字段"{type_str}"')
            if speed_kmh:
                reasons.append(f'速度{speed_kmh:.1f}km/h')
            if distance > 0:
                reasons.append(f'距离{distance:.1f}km')
            return top_sport, min(100, max_score + 20), ', '.join(reasons)

        if distance == 0 and duration > 0:
            return 'strength', 40, '零距离有时间，默认力量训练'

        return 'other', 20, '无法准确识别，标记为其他'

    def _correct_distance_duration(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()

        for idx, row in df.iterrows():
            sport = row.get('sport_type', 'running')
            distance = row.get('distance_km', 0)
            duration = row.get('duration_min', 0)
            pace = row.get('avg_pace_min_km', None)
            elevation = row.get('elevation_gain_m', None)

            distance = 0 if pd.isna(distance) else float(distance)
            duration = 0 if pd.isna(duration) else float(duration)

            corrected = False
            details = []

            if sport in ['running', 'cycling', 'walking']:
                if (distance is None or distance <= 0) and pace and duration > 0:
                    if pace > 0:
                        distance = duration / pace
                        df.at[idx, 'distance_km'] = distance
                        corrected = True
                        details.append(f'距离已推算: {distance:.2f}km (基于配速 {pace:.2f} min/km)')

                if (duration is None or duration <= 0) and pace and distance > 0:
                    duration = distance * pace
                    df.at[idx, 'duration_min'] = duration
                    corrected = True
                    details.append(f'时长已推算: {duration:.1f}min (基于配速)')

                if pace and distance > 0 and duration > 0:
                    calc_pace = duration / distance
                    if abs(calc_pace - pace) > 0.5:
                        df.at[idx, 'avg_pace_min_km'] = calc_pace
                        corrected = True
                        details.append(f'配速已校正: {calc_pace:.2f} min/km (原始:{pace:.2f})')

                if not pace and distance > 0 and duration > 0:
                    calc_pace = duration / distance
                    df.at[idx, 'avg_pace_min_km'] = calc_pace
                    corrected = True
                    details.append(f'配速已计算: {calc_pace:.2f} min/km')

                thresholds = self.reasonable_thresholds.get(sport, {})
                if thresholds.get('max_distance_single') and distance > thresholds['max_distance_single']:
                    issues.append({
                        'date': row.get('date', ''),
                        'sport_type': sport,
                        'issue_type': '异常数据警告',
                        'severity': 'warning',
                        'details': f'{sport}距离 {distance:.2f}km 超出合理范围（阈值: {thresholds["max_distance_single"]}km）'
                    })

            elif sport == 'strength':
                if distance and distance > 0:
                    df.at[idx, 'distance_km'] = 0.0
                    corrected = True
                    details.append('力量训练距离已归零')
                if not pace or pd.isna(pace):
                    df.at[idx, 'avg_pace_min_km'] = None

            elif sport == 'other':
                if not pace and distance > 0 and duration > 0:
                    df.at[idx, 'avg_pace_min_km'] = duration / distance

            if pd.isna(elevation) or elevation is None:
                df.at[idx, 'elevation_gain_m'] = 0.0

            if corrected and details:
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': sport,
                    'issue_type': '数据修正',
                    'severity': 'info',
                    'details': '; '.join(details)
                })

        return df, issues

    def _handle_missing_fields(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        issues = []
        df = df.copy()

        for idx, row in df.iterrows():
            missing = []

            if pd.isna(row.get('avg_hr')):
                missing.append('平均心率')
            if pd.isna(row.get('max_hr')):
                missing.append('最大心率')
            if pd.isna(row.get('calories')):
                missing.append('卡路里')
            if pd.isna(row.get('sleep_hours')):
                missing.append('睡眠时长')
            if pd.isna(row.get('avg_pace_min_km')) and row.get('sport_type') in ['running', 'cycling', 'walking']:
                missing.append('配速')

            if missing:
                issues.append({
                    'date': row.get('date', ''),
                    'sport_type': row.get('sport_type', ''),
                    'issue_type': '字段缺失',
                    'severity': 'info',
                    'details': f'以下字段缺失，已保留记录但跳过相关分析: {", ".join(missing)}'
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

            if pd.isna(distance):
                distance = 0
            if pd.isna(duration):
                duration = 0

            if sport in ['running', 'cycling', 'walking']:
                thresholds = self.reasonable_thresholds.get(sport, {})
                min_dist = thresholds.get('min_distance_km', 0.01)
                if distance < min_dist and duration < 1:
                    to_drop.append(idx)
                    issues.append({
                        'date': row.get('date', ''),
                        'sport_type': sport,
                        'issue_type': '无效记录',
                        'severity': 'warning',
                        'details': f'距离过短 ({distance:.3f}km) 且时长短 ({duration:.1f}min)，已删除'
                    })
            elif sport == 'strength':
                if duration < 1:
                    to_drop.append(idx)
                    issues.append({
                        'date': row.get('date', ''),
                        'sport_type': sport,
                        'issue_type': '无效记录',
                        'severity': 'warning',
                        'details': f'力量训练时长短 ({duration:.1f}min)，已删除'
                    })

        df = df.drop(index=to_drop).reset_index(drop=True)
        return df, issues

    def _ensure_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        required_fields = [
            'date', 'date_parsed', 'sport_type', 'sport_confidence',
            'distance_km', 'duration_min',
            'elevation_gain_m', 'avg_hr', 'max_hr', 'avg_pace_min_km',
            'calories', 'sleep_hours', 'injury', 'notes', 'source_file', 'track_points'
        ]
        for field in required_fields:
            if field not in df.columns:
                if field == 'date_parsed':
                    continue
                if field == 'track_points':
                    df[field] = [[] for _ in range(len(df))]
                elif field == 'sport_confidence':
                    df[field] = 100
                else:
                    df[field] = None

        numeric_fields = ['distance_km', 'duration_min', 'elevation_gain_m',
                          'avg_hr', 'max_hr', 'avg_pace_min_km', 'calories',
                          'sleep_hours', 'sport_confidence']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')

        return df
