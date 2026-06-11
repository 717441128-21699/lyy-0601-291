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

def test_outlier_detection():
    """测试异常值检测与修正"""
    print("\n" + "=" * 60)
    print("测试 7: 异常值检测与修正")
    print("=" * 60)

    gen = SampleDataGenerator(seed=42)
    raw_df = gen.generate(days=30)

    outlier_data = [
        {
            'date': '2026-06-15 07:00:00',
            'sport_type': 'running',
            'distance_km': 200,
            'duration_min': 60,
            'avg_hr': 150,
            'max_hr': 165,
            'elevation_gain_m': 50,
            'calories': 500,
            'sleep_hours': 7,
            'injury': None,
            'notes': '异常距离测试'
        },
        {
            'date': '2026-06-16 18:00:00',
            'sport_type': 'running',
            'distance_km': 5,
            'duration_min': 30,
            'avg_hr': 280,
            'max_hr': 300,
            'elevation_gain_m': 30,
            'calories': 400,
            'sleep_hours': 6.5,
            'injury': None,
            'notes': '异常心率测试'
        },
        {
            'date': '2026-06-17 12:00:00',
            'sport_type': 'strength',
            'distance_km': 0,
            'duration_min': 500,
            'avg_hr': 120,
            'max_hr': 140,
            'elevation_gain_m': 0,
            'calories': 300,
            'sleep_hours': 7,
            'injury': None,
            'notes': '异常时长测试'
        }
    ]

    outlier_df = pd.DataFrame(outlier_data)
    raw_df = pd.concat([raw_df, outlier_df], ignore_index=True)

    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(raw_df)
    clean_df, outlier_issues = cleaner.detect_outliers_for_preview(clean_df)

    print(f"✓ 总记录数: {len(clean_df)} 条")
    outlier_count = clean_df['is_outlier'].sum()
    print(f"✓ 检测到异常记录: {outlier_count} 条")
    print(f"✓ 异常问题数: {len(outlier_issues)} 个")

    for issue in outlier_issues:
        print(f"  - {issue['date']} ({issue['sport_type']}): {issue['details']}")

    clean_df_no_outliers = clean_df[~clean_df['is_outlier']].reset_index(drop=True)
    print(f"\n✓ 排除异常后: {len(clean_df_no_outliers)} 条")

    pace_analyzer = PaceAnalyzer()
    _, pace_result = pace_analyzer.analyze(clean_df_no_outliers.drop(columns=['is_outlier']))
    print(f"✓ 排除异常后训练负荷: {pace_result['summary'].get('total_training_load', 0):.0f}")

    return clean_df, outlier_issues

def test_column_mapping():
    """测试CSV列名匹配功能"""
    print("\n" + "=" * 60)
    print("测试 8: CSV列名匹配功能")
    print("=" * 60)

    gen = SampleDataGenerator(seed=42)
    raw_df = gen.generate(days=14)

    custom_col_df = raw_df[[
        'date', 'sport_type', 'distance_km', 'duration_min',
        'avg_hr', 'max_hr', 'elevation_gain_m', 'calories',
        'sleep_hours', 'injury', 'notes'
    ]].copy()

    custom_col_df.columns = [
        '运动时间', '项目', '总距离(km)', '持续时间(分)',
        '平均心率', '最大心率', '爬升(m)', '消耗卡路里',
        '睡眠时长', '伤痛情况', '备注'
    ]

    print(f"✓ 原始列名: {list(custom_col_df.columns)}")

    importer = DataImporter()
    auto_mapping = importer.auto_detect_mapping(list(custom_col_df.columns))

    print(f"✓ 自动匹配结果:")
    for key, val in auto_mapping.items():
        print(f"  {key} <-- {val}")

    assert auto_mapping.get('date') == '运动时间', "日期列匹配失败"
    assert auto_mapping.get('distance_km') == '总距离(km)', "距离列匹配失败"
    assert auto_mapping.get('duration_min') == '持续时间(分)', "时长列匹配失败"

    mapped_df = importer.apply_column_mapping(custom_col_df, auto_mapping)

    print(f"\n✓ 映射后标准列: {list(mapped_df.columns)}")
    assert 'date' in mapped_df.columns, "映射后缺少date列"
    assert 'distance_km' in mapped_df.columns, "映射后缺少distance_km列"
    print(f"✓ 映射后数据行数: {len(mapped_df)}")

    cleaner = ActivityCleaner()
    clean_df, _ = cleaner.clean_data(mapped_df)
    print(f"✓ 清洗后数据行数: {len(clean_df)}")

    return mapped_df, auto_mapping

def test_monthly_trend():
    """测试月度趋势分析功能"""
    print("\n" + "=" * 60)
    print("测试 9: 月度趋势分析（近8周训练负荷）")
    print("=" * 60)

    gen = SampleDataGenerator(seed=42)
    raw_df = gen.generate(days=70)

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

    monthly_trend = report.get('monthly_trend', {})
    print(f"✓ 趋势周数: {len(monthly_trend.get('weeks', []))} 周")
    print(f"✓ 趋势判断: {monthly_trend.get('trend_direction', '')}")
    print(f"✓ 趋势说明: {monthly_trend.get('trend_text', '')}")

    weeks = monthly_trend.get('weeks', [])
    if weeks:
        print(f"\n✓ 各周训练负荷:")
        for w in weeks:
            print(f"  {w['week_label']}: 负荷={w.get('total_load', 0):.0f}, "
                  f"时长={w.get('total_duration_min', 0):.0f}min, "
                  f"睡眠={w.get('avg_sleep', 0):.1f}h"
                  f"{' ⚠️伤痛' if w.get('has_injury') else ''}")

    summary_export = report.get('summary_export', pd.DataFrame())
    print(f"\n✓ 周报汇总导出: {len(summary_export)} 行, {len(summary_export.columns)} 列")
    if not summary_export.empty:
        print(f"  列名: {list(summary_export.columns)}")

    md = report.get('markdown_report', '')
    assert '月度趋势' in md, "Markdown报告中缺少月度趋势部分"
    assert '近8周' in md, "Markdown报告中缺少近8周表格"
    print(f"✓ Markdown包含月度趋势部分")

    return monthly_trend, report

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

    try:
        test_outlier_detection()
        results.append(("异常值检测", True))
    except Exception as e:
        results.append(("异常值检测", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_column_mapping()
        results.append(("CSV列名匹配", True))
    except Exception as e:
        results.append(("CSV列名匹配", False))
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_monthly_trend()
        results.append(("月度趋势分析", True))
    except Exception as e:
        results.append(("月度趋势分析", False))
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
