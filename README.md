# Electricity-Demand-Forecasting-Challenge-
Electricity Demand Forecasting Challenge（电力需求预测挑战赛）
首先了解赛题的背景、任务、数据以及评价规则（2024 iFLYTEK A.I.开发者大赛-讯飞开放平台）
一、赛题背景
随着全球经济的快速发展和城市化进程的加速，电力系统面临着越来越大的挑战。电力需求的准确预测对于电网的稳定运行、能源的有效管理以及可再生能源的整合至关重要。然而，电力需求受到多种因素的影响，为了提高电力需求预测的准确性和可靠性，推动智能电网和可持续能源系统的发展,本场以"电力需求预测"为赛题的数据算法挑战赛。选手需要根据历史数据构建有效的模型，能够准确的预测未来电力需求。
二、赛题任务
给定多个房屋对应电力消耗历史N天的相关序列数据等信息，预测房屋对应电力的消耗。
三、评审规则
1.数据说明
赛题数据由训练集和测试集组成，为了保证比赛的公平性，将每日日期进行脱敏，用1-N进行标识，即1为数据集最近一天，其中1-10为测试集数据。数据集由字段id（房屋id）、 dt（日标识）、type（房屋类型）、target（实际电力消耗）组成。
特征字段
字段描述
id
房屋id
dt
日标识
type
房屋类型
target
实际电力消耗，预测目标

