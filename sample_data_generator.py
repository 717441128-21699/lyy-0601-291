import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class SampleDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, days: int = 60) -> pd.DataFrame:
        today = datetime.now()
        records = []
        base_date = today - timedelta(days=days)

        current_fitness = 1.0

        for i in range(days + 1):
            date = base_date + timedelta(days=i)
            day_of_week = date.weekday()

            is_rest = random.random() < 0.15
            if is_rest and i > 0:
                sleep_h = round(random.uniform(7.5, 9.0), 1)
                records.append(self._make_record(
                    date=date,
                    sport_type='running',
                    distance=0,
                    duration=0,
                    sleep_hours=sleep_h,
                    notes='休息日'
                ))
                continue

            if day_of_week == 6:
                activity = self._generate_long_run(date, current_fitness)
            elif day_of_week == 2:
                activity = self._generate_speed_workout(date, current_fitness)
            elif day_of_week == 5:
                activity = self._generate_strength_training(date)
            else:
                r = random.random()
                if r < 0.15:
                    activity = self._generate_cycling(date, current_fitness)
                elif r < 0.25:
                    activity = self._generate_strength_training(date)
                else:
                    activity = self._generate_easy_run(date, current_fitness)

            sleep_h = round(random.uniform(5.5, 8.5), 1)
            injury = ''
            if random.random() < 0.05:
                injuries = ['膝盖轻微不适', '小腿酸痛', '跟腱轻微疼痛', '胫骨疲劳']
                injury = random.choice(injuries)

            activity['sleep_hours'] = sleep_h
            activity['injury'] = injury
            records.append(activity)

            if current_fitness < 1.3 and random.random() < 0.1:
                current_fitness += 0.01

        df = pd.DataFrame(records)
        df = df[~((df['sport_type'] == 'running') & (df['distance_km'] == 0) & (df['duration_min'] == 0))].reset_index(drop=True)
        return df

    def _make_record(self, date, sport_type, distance, duration,
                     elevation=0, avg_hr=None, max_hr=None, pace=None,
                     calories=None, sleep_hours=None, injury='', notes='',
                     source_file='sample'):
        return {
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'sport_type': sport_type,
            'distance_km': round(distance, 2),
            'duration_min': round(duration, 1),
            'elevation_gain_m': round(elevation, 1),
            'avg_hr': round(avg_hr, 1) if avg_hr else None,
            'max_hr': round(max_hr, 1) if max_hr else None,
            'avg_pace_min_km': round(pace, 2) if pace else None,
            'calories': round(calories, 0) if calories else None,
            'sleep_hours': sleep_hours,
            'injury': injury,
            'notes': notes,
            'track_points': [],
            'source_file': source_file
        }

    def _generate_easy_run(self, date, fitness):
        distance = random.uniform(5, 10) / fitness
        base_pace = random.uniform(5.8, 7.2)
        pace = base_pace / fitness
        duration = distance * pace
        avg_hr = random.uniform(125, 150)
        max_hr = avg_hr + random.uniform(5, 15)
        elevation = random.uniform(20, 120)
        calories = distance * random.uniform(60, 75)

        return self._make_record(
            date=date,
            sport_type='running',
            distance=distance,
            duration=duration,
            elevation=elevation,
            avg_hr=avg_hr,
            max_hr=max_hr,
            pace=pace,
            calories=calories,
            notes='轻松跑'
        )

    def _generate_long_run(self, date, fitness):
        distance = random.uniform(12, 22) / fitness
        base_pace = random.uniform(6.0, 7.5)
        pace = base_pace / fitness
        duration = distance * pace
        avg_hr = random.uniform(135, 155)
        max_hr = avg_hr + random.uniform(8, 18)
        elevation = random.uniform(100, 450)
        calories = distance * random.uniform(65, 80)

        return self._make_record(
            date=date,
            sport_type='running',
            distance=distance,
            duration=duration,
            elevation=elevation,
            avg_hr=avg_hr,
            max_hr=max_hr,
            pace=pace,
            calories=calories,
            notes='长距离慢跑'
        )

    def _generate_speed_workout(self, date, fitness):
        distance = random.uniform(6, 12) / fitness
        base_pace = random.uniform(4.5, 5.8)
        pace = base_pace / fitness
        duration = distance * pace
        avg_hr = random.uniform(155, 175)
        max_hr = avg_hr + random.uniform(10, 20)
        elevation = random.uniform(10, 60)
        calories = distance * random.uniform(70, 90)

        return self._make_record(
            date=date,
            sport_type='running',
            distance=distance,
            duration=duration,
            elevation=elevation,
            avg_hr=avg_hr,
            max_hr=max_hr,
            pace=pace,
            calories=calories,
            notes='速度训练/间歇'
        )

    def _generate_cycling(self, date, fitness):
        distance = random.uniform(20, 60)
        pace = random.uniform(5.0, 10.0)
        duration = distance * pace / 60 * 60
        duration_min = duration
        avg_hr = random.uniform(115, 145)
        max_hr = avg_hr + random.uniform(10, 20)
        elevation = random.uniform(100, 600)
        calories = distance * random.uniform(25, 40)

        return self._make_record(
            date=date,
            sport_type='cycling',
            distance=distance,
            duration=duration_min,
            elevation=elevation,
            avg_hr=avg_hr,
            max_hr=max_hr,
            pace=None,
            calories=calories,
            notes='骑行训练'
        )

    def _generate_strength_training(self, date):
        duration = random.uniform(30, 75)
        avg_hr = random.uniform(95, 130)
        max_hr = avg_hr + random.uniform(15, 30)
        calories = duration * random.uniform(4, 7)

        return self._make_record(
            date=date,
            sport_type='strength',
            distance=0,
            duration=duration,
            elevation=0,
            avg_hr=avg_hr,
            max_hr=max_hr,
            pace=None,
            calories=calories,
            notes='力量训练'
        )

    def export_sample_csv(self, filepath: str, days: int = 60):
        df = self.generate(days=days)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return df
