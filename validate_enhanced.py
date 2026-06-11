import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sample_data_generator import SampleDataGenerator
from modules import (
    DataImporter,
    ActivityCleaner,
    PaceAnalyzer,
    HeartRateZones,
    RecoveryReminder,
    GoalTracker,
    WeeklyReport
)

def test_full_dataset():
    """测试完整数据集（跑步+骑行+力量）"""
    print("\n" + "=" * 60)
    print("测试 1: 完整数据集（跑步+骑行+力量）")
    print("=" * 60)
    
    gen = SampleDataGenerator(seed=42)
    raw_df = gen.generate(days=60)
    
    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(raw_df)
    
    print(f"✓ 原始记录: {len(raw_df)} 条")
    print(f"✓ 清洗后记录: {len(clean_df)} 条")
    print(f"✓ 运动类型分布:")
    for sport in clean_df['sport_type'].unique():
        count = len(clean_df[clean_df['sport_type'] == sport])
        print(f"    - {sport}: {count} 条")
    
    pace_analyzer = PaceAnalyzer(resting_hr=60, max_hr=185)
    clean_df, pace_result = pace_analyzer.analyze(clean_df)
    print(f"✓ 配速分析 - 总训练负荷: {pace_result['summary'].get('total_training_load', 0):.0f}")
    
    hr_zones = HeartRateZones(resting_hr=60, max_hr=185, age=35)
    clean_df, hr_result = hr_zones.analyze(clean_df)
    print(f"✓ 心率区间 - 有心率数据: {hr_result.get('total_with_hr', 0)} 条")
    
    recovery = RecoveryReminder()
    clean_df, recovery_result = recovery.analyze(clean_df)
    print(f"✓ 恢复评分: {recovery_result['recovery_analysis'].get('recovery_score', 0)}/100")
    
    goals = GoalTracker(monthly_distance_goal_km=150, weekly_distance_goal_km=40)
    goal_result = goals.analyze(clean_df)
    print(f"✓ 跑量目标 - 仅跑步计入: {goal_result['weekly_goal'].get('progress_pct', 0):.1f}%")
    
    reporter = WeeklyReport()
    report = reporter.generate(clean_df, 
                                recovery_analysis=recovery_result,
                                goal_analysis=goal_result)
    print(f"✓ 周报生成成功")
    print(f"✓ Markdown报告长度: {len(report.get('markdown_report', ''))} 字符")
    print(f"✓ 导出数据行数: {len(report.get('export_data', pd.DataFrame()))} 行")
    print(f"✓ 恢复组合视图: {bool(report.get('recovery_combo'))}")
    
    return clean_df, report

def test_cycling_only():
    """测试只有骑行数据的场景"""
    print("\n" + "=" * 60)
    print("测试 2: 只有骑行数据（验证非跑步兼容性）")
    print("=" * 60)
    
    gen = SampleDataGenerator(seed=123)
    raw_df = gen.generate(days=30)
    
    cycling_df = raw_df[raw_df['sport_type'] == 'cycling'].copy()
    if len(cycling_df) == 0:
        cycling_df = raw_df.copy()
        cycling_df['sport_type'] = 'cycling'
        cycling_df['distance_km'] = cycling_df['distance_km'] * 3
        cycling_df['avg_pace_min_km'] = cycling_df['avg_pace_min_km'] / 3
    
    print(f"✓ 骑行记录: {len(cycling_df)} 条")
    
    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(cycling_df)
    
    print(f"✓ 清洗后: {len(clean_df)} 条, 问题: {len(issues_df)} 个")
    
    pace_analyzer = PaceAnalyzer(resting_hr=60, max_hr=185)
    clean_df, pace_result = pace_analyzer.analyze(clean_df)
    summary = pace_result.get('summary', {})
    print(f"✓ 配速分析 - has_data: {summary.get('has_data', True)}")
    print(f"✓ 按运动类型统计: {list(summary.get('by_sport', {}).keys())}")
    
    hr_zones = HeartRateZones(resting_hr=60, max_hr=185, age=35)
    clean_df, hr_result = hr_zones.analyze(clean_df)
    print(f"✓ 心率区间 - has_hr_data: {hr_result.get('has_hr_data', False)}")
    
    recovery = RecoveryReminder()
    clean_df, recovery_result = recovery.analyze(clean_df)
    rec = recovery_result.get('recovery_analysis', {})
    print(f"✓ 恢复分析 - has_data: {rec.get('has_data', True)}")
    print(f"✓ 恢复评分: {rec.get('recovery_score', 0)}/100")
    
    goals = GoalTracker(monthly_distance_goal_km=150, weekly_distance_goal_km=40)
    goal_result = goals.analyze(clean_df)
    print(f"✓ 目标追踪 - 跑步数据: {goal_result.get('has_running_data', False)}")
    print(f"✓ 所有运动总览: {bool(goal_result.get('all_sports_summary'))}")
    
    reporter = WeeklyReport()
    report = reporter.generate(clean_df, recovery_analysis=recovery_result)
    print(f"✓ 周报 - has_data: {report.get('has_data', False)}")
    print(f"✓ 周报图表数: {len(report.get('charts', {}))}")
    
    return clean_df, report

def test_strength_only():
    """测试只有力量训练数据的场景（无距离无配速）"""
    print("\n" + "=" * 60)
    print("测试 3: 只有力量训练（无距离/配速验证容错）")
    print("=" * 60)
    
    dates = pd.date_range('2025-01-01', periods=20, freq='D')
    strength_data = []
    for i, d in enumerate(dates):
        strength_data.append({
            'date': d.strftime('%Y-%m-%d %H:%M:%S'),
            'sport_type': 'strength',
            'distance_km': 0,
            'duration_min': round(np.random.uniform(30, 90), 1),
            'avg_hr': round(np.random.uniform(100, 140), 1),
            'max_hr': round(np.random.uniform(130, 160), 1),
            'avg_pace_min_km': None,
            'sleep_hours': round(np.random.uniform(6, 8.5), 1),
            'injury': '' if i % 7 != 0 else '腰部轻微不适',
            'notes': '力量训练'
        })
    
    raw_df = pd.DataFrame(strength_data)
    print(f"✓ 力量训练记录: {len(raw_df)} 条")
    print(f"✓ 距离均为0, 配速均为空")
    
    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(raw_df)
    print(f"✓ 清洗后: {len(clean_df)} 条, 问题: {len(issues_df)} 个")
    if not issues_df.empty:
        print(f"✓ 问题类型: {issues_df['issue_type'].unique().tolist()}")
    
    pace_analyzer = PaceAnalyzer(resting_hr=60, max_hr=185)
    clean_df, pace_result = pace_analyzer.analyze(clean_df)
    summary = pace_result.get('summary', {})
    print(f"✓ 配速分析 - 无跑步数据时不崩溃")
    print(f"✓ 训练总负荷: {summary.get('total_training_load', 0):.0f}")
    
    hr_zones = HeartRateZones(resting_hr=60, max_hr=185, age=35)
    clean_df, hr_result = hr_zones.analyze(clean_df)
    print(f"✓ 心率区间 - 有效: {hr_result.get('total_with_hr', 0)} 条")
    
    recovery = RecoveryReminder()
    clean_df, recovery_result = recovery.analyze(clean_df)
    rec = recovery_result.get('recovery_analysis', {})
    print(f"✓ 恢复评分: {rec.get('recovery_score', 0)}/100")
    
    goals = GoalTracker()
    goal_result = goals.analyze(clean_df)
    print(f"✓ 目标追踪 - has_running_data: {goal_result.get('has_running_data', False)}")
    
    reporter = WeeklyReport()
    report = reporter.generate(clean_df, recovery_analysis=recovery_result)
    print(f"✓ 周报 - has_data: {report.get('has_data', False)}")
    
    return clean_df, report

def test_smart_sport_detection():
    """测试智能运动类型识别"""
    print("\n" + "=" * 60)
    print("测试 4: 智能运动类型识别")
    print("=" * 60)
    
    test_cases = [
        {'sport_type': None, 'distance_km': 5, 'duration_min': 30, 'notes': '晨跑', 'expected': 'running'},
        {'sport_type': None, 'distance_km': 30, 'duration_min': 60, 'notes': '', 'expected': 'cycling'},
        {'sport_type': None, 'distance_km': 0, 'duration_min': 45, 'notes': '健身房训练', 'expected': 'strength'},
        {'sport_type': None, 'distance_km': 3, 'duration_min': 45, 'notes': '散步', 'expected': 'walking'},
        {'sport_type': '', 'distance_km': 8, 'duration_min': 50, 'notes': 'jogging', 'expected': 'running'},
        {'sport_type': 'unknown', 'distance_km': 20, 'duration_min': 40, 'notes': 'bike ride', 'expected': 'cycling'},
    ]
    
    raw_data = []
    for i, tc in enumerate(test_cases):
        raw_data.append({
            'date': f'2025-01-{i+1:02d} 08:00:00',
            'sport_type': tc['sport_type'],
            'distance_km': tc['distance_km'],
            'duration_min': tc['duration_min'],
            'avg_hr': 130,
            'notes': tc['notes']
        })
    
    raw_df = pd.DataFrame(raw_data)
    
    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(raw_df)
    
    print("✓ 识别结果:")
    correct = 0
    for tc in test_cases:
        note_match = clean_df[clean_df['notes'] == tc['notes']]
        if len(note_match) > 0:
            detected = note_match.iloc[0]['sport_type']
            conf = note_match.iloc[0].get('sport_confidence', 0)
        else:
            detected = 'unknown'
            conf = 0
        status = "✓" if detected == tc['expected'] else "✗"
        if detected == tc['expected']:
            correct += 1
        print(f"  {status} {tc['notes'] or '(无备注)'} "
              f"-> 识别为 {detected} (置信度 {conf:.0f}%) "
              f"[期望: {tc['expected']}]")
    
    print(f"✓ 准确率: {correct}/{len(test_cases)}")
    
    return clean_df, issues_df

def test_error_tolerance():
    """测试清洗容错 - 缺失字段、异常值等"""
    print("\n" + "=" * 60)
    print("测试 5: 清洗容错（缺配速/空心率/未知类型）")
    print("=" * 60)
    
    messy_data = [
        {'date': '2025-01-01 08:00:00', 'sport_type': 'running', 'distance_km': 5, 'duration_min': 30, 
         'avg_hr': None, 'avg_pace_min_km': None, 'notes': '无心率无配速'},
        {'date': '2025-01-02 08:00:00', 'sport_type': '', 'distance_km': None, 'duration_min': 45,
         'avg_hr': 130, 'notes': '未知类型无距离'},
        {'date': '2025-01-03 08:00:00', 'sport_type': 'running', 'distance_km': -1, 'duration_min': 30,
         'avg_hr': 140, 'notes': '距离为负'},
        {'date': '2025-01-04 08:00:00', 'sport_type': 'strength', 'distance_km': 0, 'duration_min': 0,
         'avg_hr': 0, 'notes': '时长为0'},
        {'date': '2025-01-05 08:00:00', 'sport_type': 'weird_sport', 'distance_km': 10, 'duration_min': 50,
         'avg_hr': 120, 'notes': '未知运动类型'},
    ]
    
    raw_df = pd.DataFrame(messy_data)
    print(f"✓ 输入脏数据: {len(raw_df)} 条")
    
    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(raw_df)
    
    print(f"✓ 清洗后保留: {len(clean_df)} 条")
    print(f"✓ 发现问题: {len(issues_df)} 个")
    
    if not issues_df.empty:
        print("\n问题明细:")
        for _, issue in issues_df.iterrows():
            print(f"  - [{issue['severity']}] {issue['issue_type']}: {issue['details']}")
    
    pace_analyzer = PaceAnalyzer(resting_hr=60, max_hr=185)
    clean_df, pace_result = pace_analyzer.analyze(clean_df)
    print(f"\n✓ 配速分析完成 - 不中断")
    
    hr_zones = HeartRateZones(resting_hr=60, max_hr=185, age=35)
    clean_df, hr_result = hr_zones.analyze(clean_df)
    print(f"✓ 心率区间分析完成 - 不中断")
    
    return clean_df, issues_df

def test_export_functions():
    """测试导出功能"""
    print("\n" + "=" * 60)
    print("测试 6: 导出功能（Markdown + CSV）")
    print("=" * 60)
    
    gen = SampleDataGenerator(seed=99)
    raw_df = gen.generate(days=30)
    
    cleaner = ActivityCleaner()
    clean_df, _ = cleaner.clean_data(raw_df)
    
    pace_analyzer = PaceAnalyzer()
    clean_df, pace_result = pace_analyzer.analyze(clean_df)
    
    hr_zones = HeartRateZones()
    clean_df, hr_result = hr_zones.analyze(clean_df)
    
    recovery = RecoveryReminder()
    clean_df, recovery_result = recovery.analyze(clean_df)
    
    goals = GoalTracker()
    goal_result = goals.analyze(clean_df)
    
    reporter = WeeklyReport()
    report = reporter.generate(clean_df, 
                                recovery_analysis=recovery_result,
                                goal_analysis=goal_result)
    
    md = report.get('markdown_report', '')
    print(f"✓ Markdown报告: {len(md)} 字符")
    if md:
        print("  预览前300字:")
        print("  " + md[:300].replace('\n', '\n  ') + "...")
    
    export_df = report.get('export_data', pd.DataFrame())
    print(f"\n✓ CSV导出数据: {len(export_df)} 行, {len(export_df.columns)} 列")
    if not export_df.empty:
        print(f"  列名: {list(export_df.columns)[:8]}...")
    
    return report

def main():
    print("=" * 60)
    print("健康管理工具 - 增强版功能验证")
    print("=" * 60)
    
    results = []
    
    try:
        test_full_dataset()
        results.append(("完整数据集", True))
    except Exception as e:
        results.append(("完整数据集", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_cycling_only()
        results.append(("只有骑行数据", True))
    except Exception as e:
        results.append(("只有骑行数据", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_strength_only()
        results.append(("只有力量训练", True))
    except Exception as e:
        results.append(("只有力量训练", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_smart_sport_detection()
        results.append(("智能运动识别", True))
    except Exception as e:
        results.append(("智能运动识别", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_error_tolerance()
        results.append(("清洗容错", True))
    except Exception as e:
        results.append(("清洗容错", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_export_functions()
        results.append(("导出功能", True))
    except Exception as e:
        results.append(("导出功能", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {status}: {name}")
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有增强功能验证通过！")
    else:
        print(f"\n⚠️  {total - passed} 项测试失败，请检查")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
