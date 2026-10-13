最小风险引擎：本地批处理；合成波动率，仅用于架构验证。

从项目目录执行（python需安装numpy和pandas）：
  python run_risk_engine.py

指定配置和输出位置：
  python run_risk_engine.py --config Input_data/risk_engine_config.json --output Output/my_run

运行自动测试：
  python -m unittest discover -s tests -v

正数量代表做多，负数量代表做空。每个合约填写一条净头寸。
当前仅支持2024-04-05至2024-04-08；其他日期会明确拒绝。
入口读取已保存的EIA市场数据与校准结果，不会重新校准。
成功退出码为0，失败返回非零；可供未来CI或任务调度使用。
测试使用临时目录，不覆盖正常运行结果。
CI配置：.github/workflows/risk-engine-ci.yml（GitHub Actions）。
推送、Pull Request或手动触发后，在Ubuntu、Python 3.12环境安装
requirements-risk.txt并运行上述四项测试，包含1000个情景的端到端计算。
不重新校准，不运行10万情景的完整作业，不需要密钥或外部数据服务。
必须提交两个现有测试输入，.gitignore已保留它们：
  Output/Step1a_EIA_Preparation/EIA_WTI_calibration_input.csv
  Output/Step1c_EIA_Full_Sample/EIA_calibration_results.csv
其余Output运行产物默认不提交；现有本地文件不会被删除。

本地安装依赖：python -m pip install -r requirements-risk.txt
仓库：qwu523isu/Commodity_future_calibration_Gabillon。
推送后在Actions页面查看Risk Engine CI；本地通过不等于云端已通过。
如需禁止未通过测试的PR合并，需在GitHub分支保护或规则集中要求
“Python 3.12 risk tests”检查通过；工作流本身不会自动启用合并限制。
目前只配置CI，未部署CD、API或服务器。
