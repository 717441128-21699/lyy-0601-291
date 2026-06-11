import os
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd

try:
    import gpxpy
except ImportError:
    gpxpy = None

try:
    from fitparse import FitFile
except ImportError:
    FitFile = None


class DataImporter:
    def __init__(self):
        self.supported_formats = ['.gpx', '.tcx', '.csv', '.fit', '.json']

        self.standard_fields = [
            {'key': 'date', 'label': '日期/开始时间', 'required': True,
             'aliases': ['date', 'Date', 'timestamp', 'Timestamp', 'start_time', 'StartTime', '开始时间', '日期', '时间', '运动时间', '训练时间', '运动日期']},
            {'key': 'sport_type', 'label': '运动类型', 'required': False,
             'aliases': ['sport_type', 'type', 'sport', 'Sport', 'Type', 'activity_type', 'Activity', '运动类型', '类型', '项目']},
            {'key': 'distance_km', 'label': '距离(km)', 'required': False,
             'aliases': ['distance_km', 'distance', 'Distance', '距离', '里程', '总距离', '总距离(km)', '距离(公里)', '跑步距离', '运动距离']},
            {'key': 'duration_min', 'label': '时长(min)', 'required': False,
             'aliases': ['duration_min', 'duration', 'Duration', '时长', '时间', '分钟', '持续时间', '持续时间(分)', '时长(分钟)', '总时长', '运动时长', '训练时长']},
            {'key': 'elevation_gain_m', 'label': '爬升(m)', 'required': False,
             'aliases': ['elevation_gain_m', 'elevation', 'Elevation', 'climb', 'Climb', '爬升', '海拔提升', '上升', '爬升(m)', '总爬升', '海拔']},
            {'key': 'avg_hr', 'label': '平均心率', 'required': False,
             'aliases': ['avg_hr', 'average_hr', 'Avg HR', 'mean_hr', '平均心率', '心率', '平均心率(bpm)']},
            {'key': 'max_hr', 'label': '最大心率', 'required': False,
             'aliases': ['max_hr', 'Max HR', 'max_heart_rate', '最大心率', '最高心率', '最大心率(bpm)', '峰值心率']},
            {'key': 'avg_pace_min_km', 'label': '平均配速(min/km)', 'required': False,
             'aliases': ['avg_pace_min_km', 'pace', 'Pace', '配速', '平均配速', '配速(min/km)', '平均配速(min/km)']},
            {'key': 'calories', 'label': '卡路里', 'required': False,
             'aliases': ['calories', 'Calories', '卡路里', '消耗', '热量', '消耗卡路里', '消耗热量', '卡路里(kcal)']},
            {'key': 'sleep_hours', 'label': '睡眠(h)', 'required': False,
             'aliases': ['sleep_hours', 'sleep', 'Sleep', '睡眠', '睡眠时间', '睡眠时长', '睡眠(h)', '睡眠(小时)']},
            {'key': 'injury', 'label': '伤痛', 'required': False,
             'aliases': ['injury', 'pain', 'Injury', '伤痛', '受伤', '疼痛', '伤痛情况', '是否受伤']},
            {'key': 'notes', 'label': '备注', 'required': False,
             'aliases': ['notes', 'Notes', 'comment', 'Comment', '备注', '说明', '描述']},
        ]

        self.field_labels = {f['key']: f['label'] for f in self.standard_fields}

    def import_files(self, file_paths: List[str]) -> pd.DataFrame:
        activities = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == '.gpx':
                    activity = self._parse_gpx(file_path)
                elif ext == '.tcx':
                    activity = self._parse_tcx(file_path)
                elif ext == '.csv':
                    activity = self._parse_csv(file_path)
                elif ext == '.fit':
                    activity = self._parse_fit(file_path)
                elif ext == '.json':
                    activity = self._parse_json(file_path)
                else:
                    continue
                if activity:
                    activity['source_file'] = os.path.basename(file_path)
                    activities.append(activity)
            except Exception as e:
                print(f"解析文件失败 {file_path}: {e}")
        return pd.DataFrame(activities)

    def _parse_gpx(self, file_path: str) -> Optional[Dict[str, Any]]:
        if not gpxpy:
            return self._parse_gpx_xml(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            activity = {
                'date': None,
                'sport_type': 'running',
                'distance_km': 0.0,
                'duration_min': 0.0,
                'elevation_gain_m': 0.0,
                'avg_hr': None,
                'max_hr': None,
                'avg_pace_min_km': None,
                'track_points': []
            }
            if gpx.tracks:
                track = gpx.tracks[0]
                activity['distance_km'] = track.length_3d() / 1000.0 if track.length_3d() else 0.0
                moving_data = track.get_moving_data()
                if moving_data:
                    activity['duration_min'] = moving_data.moving_time / 60.0 if moving_data.moving_time else 0.0
                activity['elevation_gain_m'] = track.get_uphill_downhill().uphill if track.get_uphill_downhill() else 0.0
                if track.segments:
                    for segment in track.segments:
                        for point in segment.points:
                            tp = {
                                'lat': point.latitude,
                                'lon': point.longitude,
                                'ele': point.elevation,
                                'time': point.time.isoformat() if point.time else None
                            }
                            if hasattr(point, 'extensions') and point.extensions:
                                for ext in point.extensions:
                                    for child in ext:
                                        if 'hr' in child.tag.lower() or 'heart' in child.tag.lower():
                                            tp['hr'] = int(child.text) if child.text else None
                            activity['track_points'].append(tp)
                times = [p.time for seg in track.segments for p in seg.points if p.time]
                if times:
                    activity['date'] = times[0].isoformat()
            hrs = [tp.get('hr') for tp in activity['track_points'] if tp.get('hr')]
            if hrs:
                activity['avg_hr'] = sum(hrs) / len(hrs)
                activity['max_hr'] = max(hrs)
            if activity['distance_km'] > 0 and activity['duration_min'] > 0:
                activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
            return activity
        except Exception as e:
            print(f"GPX解析错误: {e}")
            return None

    def _parse_gpx_xml(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = '{http://www.topografix.com/GPX/1/1}'
            activity = {
                'date': None,
                'sport_type': 'running',
                'distance_km': 0.0,
                'duration_min': 0.0,
                'elevation_gain_m': 0.0,
                'avg_hr': None,
                'max_hr': None,
                'avg_pace_min_km': None,
                'track_points': []
            }
            track_points = []
            for trkpt in root.iter(f'{ns}trkpt'):
                tp = {
                    'lat': float(trkpt.get('lat', 0)),
                    'lon': float(trkpt.get('lon', 0)),
                    'ele': None,
                    'time': None,
                    'hr': None
                }
                ele = trkpt.find(f'{ns}ele')
                if ele is not None and ele.text:
                    tp['ele'] = float(ele.text)
                time_el = trkpt.find(f'{ns}time')
                if time_el is not None and time_el.text:
                    tp['time'] = time_el.text
                    try:
                        dt = datetime.fromisoformat(time_el.text.replace('Z', '+00:00'))
                        tp['time_parsed'] = dt
                    except:
                        pass
                extensions = trkpt.find(f'{ns}extensions')
                if extensions is not None:
                    for ext in extensions.iter():
                        if 'hr' in ext.tag.lower() or 'heart' in ext.tag.lower():
                            if ext.text:
                                tp['hr'] = int(ext.text)
                track_points.append(tp)
            activity['track_points'] = track_points
            if len(track_points) >= 2:
                total_dist = 0.0
                prev_ele = None
                for i in range(1, len(track_points)):
                    total_dist += self._haversine(
                        track_points[i-1]['lat'], track_points[i-1]['lon'],
                        track_points[i]['lat'], track_points[i]['lon']
                    )
                    if track_points[i]['ele'] is not None and prev_ele is not None:
                        if track_points[i]['ele'] > prev_ele:
                            activity['elevation_gain_m'] += track_points[i]['ele'] - prev_ele
                    if track_points[i]['ele'] is not None:
                        prev_ele = track_points[i]['ele']
                activity['distance_km'] = total_dist / 1000.0
                times = [tp.get('time_parsed') for tp in track_points if tp.get('time_parsed')]
                if len(times) >= 2:
                    duration = (times[-1] - times[0]).total_seconds() / 60.0
                    activity['duration_min'] = duration
                    activity['date'] = times[0].isoformat()
            hrs = [tp.get('hr') for tp in track_points if tp.get('hr')]
            if hrs:
                activity['avg_hr'] = sum(hrs) / len(hrs)
                activity['max_hr'] = max(hrs)
            if activity['distance_km'] > 0 and activity['duration_min'] > 0:
                activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
            return activity
        except Exception as e:
            print(f"GPX XML解析错误: {e}")
            return None

    def _parse_tcx(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}'
            activity = {
                'date': None,
                'sport_type': 'running',
                'distance_km': 0.0,
                'duration_min': 0.0,
                'elevation_gain_m': 0.0,
                'avg_hr': None,
                'max_hr': None,
                'avg_pace_min_km': None,
                'track_points': []
            }
            activity_el = root.find(f'.//{ns}Activity')
            if activity_el is not None:
                sport = activity_el.get('Sport', 'Running').lower()
                if 'cycling' in sport or 'bike' in sport:
                    activity['sport_type'] = 'cycling'
                elif 'running' in sport:
                    activity['sport_type'] = 'running'
                else:
                    activity['sport_type'] = 'other'
            lap = root.find(f'.//{ns}Lap')
            if lap is not None:
                total_time = lap.find(f'{ns}TotalTimeSeconds')
                distance = lap.find(f'{ns}DistanceMeters')
                if total_time is not None and total_time.text:
                    activity['duration_min'] = float(total_time.text) / 60.0
                if distance is not None and distance.text:
                    activity['distance_km'] = float(distance.text) / 1000.0
                id_el = lap.find(f'.//{ns}Id')
                if id_el is not None and id_el.text:
                    activity['date'] = id_el.text
            track_points = []
            for tp_el in root.iter(f'{ns}Trackpoint'):
                tp = {'lat': None, 'lon': None, 'ele': None, 'time': None, 'hr': None}
                pos = tp_el.find(f'{ns}Position')
                if pos is not None:
                    lat = pos.find(f'{ns}LatitudeDegrees')
                    lon = pos.find(f'{ns}LongitudeDegrees')
                    if lat is not None and lat.text:
                        tp['lat'] = float(lat.text)
                    if lon is not None and lon.text:
                        tp['lon'] = float(lon.text)
                ele = tp_el.find(f'{ns}AltitudeMeters')
                if ele is not None and ele.text:
                    tp['ele'] = float(ele.text)
                time_el = tp_el.find(f'{ns}Time')
                if time_el is not None and time_el.text:
                    tp['time'] = time_el.text
                hr_el = tp_el.find(f'.//{ns}Value')
                if hr_el is not None and hr_el.text:
                    tp['hr'] = int(hr_el.text)
                track_points.append(tp)
            activity['track_points'] = track_points
            hrs = [tp.get('hr') for tp in track_points if tp.get('hr')]
            if hrs:
                activity['avg_hr'] = sum(hrs) / len(hrs)
                activity['max_hr'] = max(hrs)
            if activity['distance_km'] > 0 and activity['duration_min'] > 0:
                activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
            return activity
        except Exception as e:
            print(f"TCX解析错误: {e}")
            return None

    def _parse_csv(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            df = pd.read_csv(file_path)
            activities = []
            for _, row in df.iterrows():
                activity = {
                    'date': str(row.get('date', row.get('Date', row.get('timestamp', '')))),
                    'sport_type': str(row.get('sport_type', row.get('type', row.get('sport', 'running')))).lower(),
                    'distance_km': float(row.get('distance_km', row.get('distance', row.get('Distance', 0)))),
                    'duration_min': float(row.get('duration_min', row.get('duration', row.get('Duration', 0)))),
                    'elevation_gain_m': float(row.get('elevation_gain_m', row.get('elevation', row.get('Elevation', 0)))),
                    'avg_hr': self._safe_float(row.get('avg_hr', row.get('average_hr', row.get('Avg HR', None)))),
                    'max_hr': self._safe_float(row.get('max_hr', row.get('Max HR', None))),
                    'avg_pace_min_km': self._safe_float(row.get('avg_pace_min_km', row.get('pace', None))),
                    'calories': self._safe_float(row.get('calories', None)),
                    'sleep_hours': self._safe_float(row.get('sleep_hours', None)),
                    'injury': str(row.get('injury', row.get('pain', ''))),
                    'notes': str(row.get('notes', row.get('Notes', ''))),
                    'track_points': []
                }
                if activity['distance_km'] > 0 and activity['duration_min'] > 0 and not activity['avg_pace_min_km']:
                    activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
                activities.append(activity)
            if len(activities) == 1:
                return activities[0]
            elif len(activities) > 1:
                return None
        except Exception as e:
            print(f"CSV解析错误: {e}")
            return None
        return None

    def read_csv_raw(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"CSV读取错误: {e}")
            return None

    def auto_detect_mapping(self, df_columns: List[str]) -> Dict[str, str]:
        mapping = {}
        for field in self.standard_fields:
            for alias in field['aliases']:
                if alias in df_columns:
                    mapping[field['key']] = alias
                    break
        return mapping

    def apply_column_mapping(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()

        for std_key, src_col in column_mapping.items():
            if src_col and src_col in result.columns:
                if std_key != src_col:
                    if std_key in result.columns and std_key != src_col:
                        result = result.drop(columns=[std_key])
                    result = result.rename(columns={src_col: std_key})

        for field in self.standard_fields:
            key = field['key']
            if key not in result.columns:
                if key in ['date', 'sport_type', 'notes']:
                    result[key] = ''
                elif key == 'injury':
                    result[key] = ''
                else:
                    result[key] = None

        return result

    def _parse_fit(self, file_path: str) -> Optional[Dict[str, Any]]:
        if not FitFile:
            return None
        try:
            fit = FitFile(file_path)
            activity = {
                'date': None,
                'sport_type': 'running',
                'distance_km': 0.0,
                'duration_min': 0.0,
                'elevation_gain_m': 0.0,
                'avg_hr': None,
                'max_hr': None,
                'avg_pace_min_km': None,
                'track_points': []
            }
            hrs = []
            for record in fit.get_messages('record'):
                tp = {'lat': None, 'lon': None, 'ele': None, 'time': None, 'hr': None}
                for data in record:
                    if data.name == 'position_lat' and data.value:
                        tp['lat'] = data.value * (180.0 / 2**31)
                    elif data.name == 'position_long' and data.value:
                        tp['lon'] = data.value * (180.0 / 2**31)
                    elif data.name == 'altitude' and data.value:
                        tp['ele'] = float(data.value)
                    elif data.name == 'timestamp' and data.value:
                        tp['time'] = data.value.isoformat() if hasattr(data.value, 'isoformat') else str(data.value)
                    elif data.name == 'heart_rate' and data.value:
                        tp['hr'] = int(data.value)
                        hrs.append(int(data.value))
                activity['track_points'].append(tp)
            for session in fit.get_messages('session'):
                for data in session:
                    if data.name == 'sport' and data.value:
                        sport = str(data.value).lower()
                        if 'cycling' in sport or 'bike' in sport:
                            activity['sport_type'] = 'cycling'
                        elif 'running' in sport:
                            activity['sport_type'] = 'running'
                        else:
                            activity['sport_type'] = 'other'
                    elif data.name == 'total_distance' and data.value:
                        activity['distance_km'] = float(data.value) / 1000.0
                    elif data.name == 'total_elapsed_time' and data.value:
                        activity['duration_min'] = float(data.value) / 60.0
                    elif data.name == 'total_ascent' and data.value:
                        activity['elevation_gain_m'] = float(data.value)
                    elif data.name == 'avg_heart_rate' and data.value:
                        activity['avg_hr'] = float(data.value)
                    elif data.name == 'max_heart_rate' and data.value:
                        activity['max_hr'] = float(data.value)
                    elif data.name == 'timestamp' and data.value:
                        activity['date'] = data.value.isoformat() if hasattr(data.value, 'isoformat') else str(data.value)
            if hrs and not activity['avg_hr']:
                activity['avg_hr'] = sum(hrs) / len(hrs)
            if hrs and not activity['max_hr']:
                activity['max_hr'] = max(hrs)
            if activity['distance_km'] > 0 and activity['duration_min'] > 0:
                activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
            return activity
        except Exception as e:
            print(f"FIT解析错误: {e}")
            return None

    def _parse_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                if len(data) == 1:
                    data = data[0]
                else:
                    return None
            activity = {
                'date': str(data.get('date', data.get('timestamp', ''))),
                'sport_type': str(data.get('sport_type', data.get('type', 'running'))).lower(),
                'distance_km': float(data.get('distance_km', data.get('distance', 0))),
                'duration_min': float(data.get('duration_min', data.get('duration', 0))),
                'elevation_gain_m': float(data.get('elevation_gain_m', data.get('elevation', 0))),
                'avg_hr': self._safe_float(data.get('avg_hr', data.get('average_hr'))),
                'max_hr': self._safe_float(data.get('max_hr')),
                'avg_pace_min_km': self._safe_float(data.get('avg_pace_min_km', data.get('pace'))),
                'calories': self._safe_float(data.get('calories')),
                'sleep_hours': self._safe_float(data.get('sleep_hours')),
                'injury': str(data.get('injury', data.get('pain', ''))),
                'notes': str(data.get('notes', '')),
                'track_points': data.get('track_points', [])
            }
            if activity['distance_km'] > 0 and activity['duration_min'] > 0 and not activity['avg_pace_min_km']:
                activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
            return activity
        except Exception as e:
            print(f"JSON解析错误: {e}")
            return None

    def _safe_float(self, val) -> Optional[float]:
        if val is None or val == '' or (isinstance(val, str) and val.strip() == ''):
            return None
        try:
            return float(val)
        except:
            return None

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        import math
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def import_csv_batch(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            records = []
            for _, row in df.iterrows():
                activity = {
                    'date': str(row.get('date', row.get('Date', row.get('timestamp', '')))),
                    'sport_type': str(row.get('sport_type', row.get('type', row.get('sport', 'running')))).lower(),
                    'distance_km': self._safe_float(row.get('distance_km', row.get('distance', row.get('Distance', 0)))),
                    'duration_min': self._safe_float(row.get('duration_min', row.get('duration', row.get('Duration', 0)))),
                    'elevation_gain_m': self._safe_float(row.get('elevation_gain_m', row.get('elevation', row.get('Elevation', 0)))),
                    'avg_hr': self._safe_float(row.get('avg_hr', row.get('average_hr', row.get('Avg HR', None)))),
                    'max_hr': self._safe_float(row.get('max_hr', row.get('Max HR', None))),
                    'avg_pace_min_km': self._safe_float(row.get('avg_pace_min_km', row.get('pace', None))),
                    'calories': self._safe_float(row.get('calories', None)),
                    'sleep_hours': self._safe_float(row.get('sleep_hours', None)),
                    'injury': str(row.get('injury', row.get('pain', ''))),
                    'notes': str(row.get('notes', row.get('Notes', ''))),
                    'track_points': [],
                    'source_file': os.path.basename(file_path)
                }
                if activity['distance_km'] and activity['duration_min'] and activity['distance_km'] > 0 and activity['duration_min'] > 0 and not activity['avg_pace_min_km']:
                    activity['avg_pace_min_km'] = activity['duration_min'] / activity['distance_km']
                records.append(activity)
            return pd.DataFrame(records)
        except Exception as e:
            print(f"批量CSV解析错误: {e}")
            return pd.DataFrame()
