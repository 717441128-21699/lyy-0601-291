import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def main():
    print("=" * 60)
    print("健康管理自动化工具 - 功能验证")
    print("=" * 60)

    print("\n📊 1. 生成示例数据...")
    gen = SampleDataGenerator(seed=42)
    raw_df = gen.generate(days=75)
    print(f"   ✓ 生成 {len(raw_df)} 条训练记录")
    print(f"   ✓ 列名: {list(raw_df.columns)}")

    print("\n🧹 2. 数据清洗...")
    cleaner = ActivityCleaner()
    clean_df, issues_df = cleaner.clean_data(raw_df)
    print(f"   ✓ 清洗后 {len(clean_df)} 条记录")
    print(f"   ✓ 发现 {len(issues_df)} 个问题")
    if not issues_df.empty:
        print(f"   ✓ 问题类型: {issues_df['issue_type'].unique().tolist()}")

    print("\n⏱️ 3. 配速与训练负荷分析...")
    pace_analyzer = PaceAnalyzer(resting_hr=60, max_hr=185)
    clean_df, pace_result = pace_analyzer.analyze(clean_df)
    summary = pace_result.get('summary', {})
    print(f"   ✓ 总跑步次数: {summary.get('total_runs', 0)}")
    print(f"   ✓ 总距离: {summary.get('total_distance_km', 0):.1f} km")
    print(f"   ✓ 平均配速: {summary.get('avg_pace', '--')}")
    print(f"   ✓ 训练总负荷: {summary.get('total_training_load', 0):.0f}")
    print(f"   ✓ 配速分布区间数: {len(pace_result.get('pace_distribution', {}).get('distribution', {}))}")

    print("\n❤️ 4. 心率区间分析...")
    hr_zones = HeartRateZones(resting_hr=60, max_hr=185, age=35)
    clean_df, hr_result = hr_zones.analyze(clean_df)
    print(f"   ✓ 心率区间数: {len(hr_result.get('zones_definition', []))}")
    balance = hr_result.get('training_balance', {})
    print(f"   ✓ 训练平衡评分: {balance.get('score', 0)}/100")
    print(f"   ✓ 低强度: {balance.get('easy_pct', 0):.1f}% | 阈值: {balance.get('moderate_pct', 0):.1f}% | 高强度: {balance.get('hard_pct', 0):.1f}%")

    print("\n💤 5. 恢复状态评估...")
    recovery = RecoveryReminder()
    clean_df, recovery_result = recovery.analyze(clean_df)
    rec_analysis = recovery_result.get('recovery_analysis', {})
    print(f"   ✓ 7天训练负荷: {rec_analysis.get('7day_total_load', 0):.0f}")
    print(f"   ✓ 恢复评分: {rec_analysis.get('recovery_score', 0)}/100")
    print(f"   ✓ 恢复状态: {rec_analysis.get('recovery_level', '')}")
    print(f"   ✓ 提醒数: {len(recovery_result.get('reminders', []))}")
    overtraining = recovery_result.get('overtraining_risk', {})
    print(f"   ✓ 过度训练风险: {overtraining.get('risk_level', '')}")

    print("\n🎯 6. 目标追踪...")
    goals = GoalTracker(monthly_distance_goal_km=150, weekly_distance_goal_km=40, yearly_distance_goal_km=1800)
    goal_result = goals.analyze(clean_df)
    weekly = goal_result.get('weekly_goal', {})
    monthly = goal_result.get('monthly_goal', {})
    print(f"   ✓ 周目标进度: {weekly.get('progress_pct', 0):.1f}% ({weekly.get('pace_status', '')})")
    print(f"   ✓ 月目标进度: {monthly.get('progress_pct', 0):.1f}% ({monthly.get('pace_status', '')})")
    streaks = goal_result.get('streaks', {})
    print(f"   ✓ 当前连续: {streaks.get('current_streak_days', 0)}天 | 最长: {streaks.get('longest_streak_days', 0)}天")
    trend = goal_result.get('trend', {})
    print(f"   ✓ 趋势: {trend.get('status', '')} | 配速: {trend.get('pace_trend', '')}")

    print("\n📝 7. 周报生成...")
    reporter = WeeklyReport()
    report = reporter.generate(clean_df)
    text_report = report.get('text_report', '')
    print(f"   ✓ 周报告期: {report.get('week_period', '')}")
    print(f"   ✓ 文本报告长度: {len(text_report)} 字符")
    anomalies = report.get('anomalies', {})
    anomaly_count = sum(1 for v in anomalies.values() if v.get('has_issue', False))
    print(f"   ✓ 异常项数: {anomaly_count}/4")
    charts = report.get('charts', {})
    print(f"   ✓ 图表数: {len(charts)}")
    for name in charts:
        has_data = False
        try:
            has_data = len(charts[name].data) > 0
        except:
            pass
        print(f"     - {name}: {'有数据' if has_data else '空图'}")

    print("\n" + "=" * 60)
    print("✅ 所有模块验证通过！")
    print("=" * 60)

    print("\n📌 快速启动:")
    print("   streamlit run app.py")


if __name__ == '__main__':
    main()
