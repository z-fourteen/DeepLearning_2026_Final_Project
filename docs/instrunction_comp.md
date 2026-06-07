实时交易策略构建
# 1. 确保参数更新
每次运行前查看北京时间，查看文件夹`D:\YangXiran\USTC\0_Math\DeepLearning\Final\DeepLearning_2026_Final_Project\A股数据`内的数据是否及时更新并更改有关脚本的enddate，没有更新请提醒开发者更新

# 2. 脚本构建
搜索文件夹内的实时交易策略脚本
适应性地修改以下脚本运行示范：


    全量一键全流程跑 00-05：
    D:\06_envs\dl_env\python.exe scripts/live/live_daily.py --run-dag --full-dag --data-version v20260526 --end-date 20260529 --trade-date 20260601 --run-intraday-monitor --execute --initial-nav 1000000 --no-push
    日常增量一键全流程跑 00-05：
    D:\06_envs\dl_env\python.exe scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260529 --trade-date 20260601 --run-intraday-monitor --execute --initial-nav 1000000 --no-push
    如果要让 04 持续循环监控，加上：
    --monitor-loop
    如果 05 要自动 commit/push，去掉：
    --no-push



# 3. 模型使用
使用三个模型：
`StdTF-l60-cls-mse-f13` K=20

`StdTF-l60-cls-mse-f13` K=10

`EnhancedTF-l20-cls-huber-f13` K=20

两个模型分别产生两个交易策略，不要互相干扰

