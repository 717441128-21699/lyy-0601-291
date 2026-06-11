import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class HeartRateZones:
    def __init__(self, resting_hr: float = 60.0, max_hr: Optional[float] = None, age: int = 35):
        self.resting_hr = resting_hr
        self.age = age
        self.max_hr = max_hr if max_hr else self._estimate_max_hr()
        self.hrr = self.max_hr - self.resting_hr
        self.zones = self._define_zones()

    def _estimate_max_hr(self) -> float:
        return 220.0 - float(self.age)

    def _define_zones(self) -> List[Dict]:
        zones = [
            {
                'name': 'Z1 恢复',
                'color': '#3498db',
                'min_pct': 0.0,
                'max_pct': 0.6,
                'min_hr': self.resting_hr,
                'max_hr': self.resting_hr + 0.6 * self.hrr,
                'description': '非常轻松的活动，用于恢复和热身'
            },
            {
                'name': 'Z2 有氧',
                'color': '#2ecc71',
                'min_pct': 0.6,
                'max_pct': 0.7,
                'min_hr': self.resting_hr + 0.6 * self.hrr,
                'max_hr': self.resting_hr + 0.7 * self.hrr,
                'description': '舒适的有氧配速，可以边跑边聊天'
            },
            {
                'name': 'Z3 阈值',
                'color': '#f1c40f',
                'min_pct': 0.7,
                'max_pct': 0.8,
                'min_hr': self.resting_hr + 0.7 * self.hrr,
                'max_hr': self.resting_hr + 0.8 * self.hrr,
                'description': '中等强度，说话有些困难'
            },
            {
                'name': 'Z4 无氧',
                'color': '#e67e22',
                'min_pct': 0.8,
                'max_pct': 0.9,
                'min_hr': self.resting_hr + 0.8 * self.hrr,
                'max_hr': self.resting_hr + 0.9 * self.hrr,
                'description': '高强度，只能说短句'
            },
            {
                'name': 'Z5 极限',
                'color': '#e74c3c',
                'min_pct': 0.9,
                'max_pct': 1.0,
                'min_hr': self.resting_hr + 0.9 * self.hrr,
                'max_hr': self.max_hr,
                'description': '极限强度，只能坚持很短时间'
            }
        ]
        return zones

    def get_zone_for_hr(self, hr: float) -> Optional[Dict]:
        if not hr:
            return None
        for zone in self.zones:
            if zone['min_hr'] <= hr < zone['max_hr']:
                return zone
        if hr >= self.zones[-1]['min_hr']:
            return self.zones[-1]
        return None

    def get_zone_name(self, hr: float) -> str:
        zone = self.get_zone_for_hr(hr)
        return zone['name'] if zone else '未知'

    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        if df.empty:
            return df, {}

        df = df.copy()

        df['hr_zone'] = df['avg_hr'].apply(self.get_zone_name)

        zone_stats = self._calculate_zone_stats(df)
        zone_distribution = self._zone_distribution(df)
        training_balance = self._assess_training_balance(df)

        analysis = {
            'zones_definition': self.zones,
            'zone_stats': zone_stats,
            'zone_distribution': zone_distribution,
            'training_balance': training_balance,
            'resting_hr': self.resting_hr,
            'max_hr': self.max_hr,
            'hrr': round(self.hrr, 1)
        }

        return df, analysis

    def _calculate_zone_stats(self, df: pd.DataFrame) -> Dict:
        stats = {}
        for zone in self.zones:
            zone_name = zone['name']
            zone_df = df[df['hr_zone'] == zone_name]
            if len(zone_df) > 0:
                stats[zone_name] = {
                    'count': len(zone_df),
                    'total_duration_min': round(zone_df['duration_min'].sum(), 1),
                    'avg_duration_min': round(zone_df['duration_min'].mean(), 1),
                    'total_distance_km': round(zone_df['distance_km'].sum(), 2),
                    'avg_hr': round(zone_df['avg_hr'].mean(), 1) if zone_df['avg_hr'].notna().any() else None,
                    'total_load': round(zone_df['training_load'].sum(), 1) if 'training_load' in zone_df.columns else 0
                }
            else:
                stats[zone_name] = {
                    'count': 0,
                    'total_duration_min': 0,
                    'avg_duration_min': 0,
                    'total_distance_km': 0,
                    'avg_hr': None,
                    'total_load': 0
                }
        return stats

    def _zone_distribution(self, df: pd.DataFrame) -> Dict:
        zone_counts = df['hr_zone'].value_counts()
        total = len(df)
        distribution = {}
        for zone in self.zones:
            count = int(zone_counts.get(zone['name'], 0))
            distribution[zone['name']] = {
                'count': count,
                'percentage': round(count / total * 100, 1) if total > 0 else 0,
                'color': zone['color']
            }
        other_count = total - sum([v['count'] for v in distribution.values()])
        if other_count > 0:
            distribution['未知/无心率'] = {
                'count': other_count,
                'percentage': round(other_count / total * 100, 1),
                'color': '#95a5a6'
            }
        return distribution

    def _assess_training_balance(self, df: pd.DataFrame) -> Dict:
        df_with_hr = df[df['avg_hr'].notna()]
        if df_with_hr.empty:
            return {
                'assessment': '暂无足够心率数据',
                'recommendations': ['建议佩戴心率设备记录训练'],
                'score': 0
            }

        zone_counts = df_with_hr['hr_zone'].value_counts()
        total = len(df_with_hr)

        z1_pct = zone_counts.get('Z1 恢复', 0) / total * 100
        z2_pct = zone_counts.get('Z2 有氧', 0) / total * 100
        z3_pct = zone_counts.get('Z3 阈值', 0) / total * 100
        z4_pct = zone_counts.get('Z4 无氧', 0) / total * 100
        z5_pct = zone_counts.get('Z5 极限', 0) / total * 100

        easy_pct = z1_pct + z2_pct
        moderate_pct = z3_pct
        hard_pct = z4_pct + z5_pct

        recommendations = []
        score = 0

        if easy_pct >= 70 and easy_pct <= 85:
            score += 40
            recommendations.append('有氧基础训练比例合理（70-85%）')
        elif easy_pct < 70:
            recommendations.append(f'低强度有氧训练比例偏低（当前{easy_pct:.1f}%），建议增加Z1-Z2区间训练至70%以上')
        elif easy_pct > 85:
            recommendations.append(f'低强度训练比例过高（当前{easy_pct:.1f}%），可适当增加一些高强度训练')

        if hard_pct >= 5 and hard_pct <= 20:
            score += 30
            recommendations.append('高强度训练比例合理（5-20%）')
        elif hard_pct > 20:
            recommendations.append(f'高强度训练比例过高（当前{hard_pct:.1f}%），注意恢复，避免过度训练')
            score += 10
        elif hard_pct < 5 and total >= 5:
            recommendations.append(f'高强度训练较少（当前{hard_pct:.1f}%），可适当加入间歇训练')
            score += 15

        if moderate_pct >= 10 and moderate_pct <= 20:
            score += 30
            recommendations.append('阈值训练比例合理')
        elif moderate_pct > 30:
            recommendations.append(f'阈值区间训练过多（当前{moderate_pct:.1f}%），俗称"垃圾里程"，建议调整到更高或更低强度')

        score = min(100, score)

        return {
            'easy_pct': round(easy_pct, 1),
            'moderate_pct': round(moderate_pct, 1),
            'hard_pct': round(hard_pct, 1),
            'assessment': f'训练平衡评分: {score}/100' if total >= 3 else '数据较少，建议积累更多训练记录',
            'recommendations': recommendations if recommendations else ['训练分布正常'],
            'score': score
        }
