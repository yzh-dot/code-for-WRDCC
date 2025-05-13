import pandas as pd
import matplotlib.pyplot as plt
from rsome import grb_solver as grb
from rsome import cpt_solver as cpt
from rsome import dro
from rsome import ro
from rsome import E
import rsome as rso
import time
import numpy as np
np.set_printoptions(threshold=np.inf)

## 1. 整体数据

# 获取原始elia风力发电数据（24 hours）
winddata = pd.read_excel('data\\2016-2023-24hours-elia.xlsx')

# 最近预测风电数据
wind_recent_forecast = winddata['Most recent forecast [MW]'].values

# 日前预测风电数据（11h00）
wind_dayahead_forecast = winddata['Day-ahead forecast (11h00) [MW]'].values

# 风机容量（逐年增长）
wind_capacity = winddata['Monitored Capacity [MW]'].values

#实际风电数据
wind_measured = winddata['Measured & upscaled [MW]'].values

# 将日前预测风电数据归一化
wind_dayahead_forecast_Normalization = wind_dayahead_forecast/wind_capacity

# 将风电数据改成一天一行的形式，便于DTC训练
reshaped_data = wind_dayahead_forecast_Normalization.reshape(-1, 24)


'''
# 创建DataFrame
df = pd.DataFrame({'WindForecast': wind_dayahead_forecast, 'windTrue': wind_measured, 'Monitored Capacity': wind_capacity})

# 保存到xlsx文件（用于DRO 30 Bus调度）
df.to_excel('data\error_elia.xlsx', index=False)
'''

'''
## 2. 保存数据到相应的文件夹里

# 将日前预测风电数据归一化
wind_dayahead_forecast_Normalization = (wind_dayahead_forecast)/wind_capacity

# 将风电数据改成一天一行的形式，便于DTC训练
reshaped_data = wind_dayahead_forecast_Normalization.reshape(-1, 24)

# 创建新的DataFrame
new_df = pd.DataFrame(reshaped_data)

# 写入到新的Excel文件（用于DTC训练）
new_df.to_excel(r'C:\\Users\\余志航\\Desktop\\RBC和电力调度\\代码\\Deep-temporal-clustering\\windforecast.xlsx', index=False, header=False)


## 3. DTC训练完成后数据处理(预测数据分类)

# 获取带有标签的数据
winddata = pd.read_csv(r"data\results_data.csv")

# 统计每种标签的数量
label_counts = winddata['Label'].value_counts()

# 找出数量最少的 n 种标签
n = 10  # 找出数量最少的 n 种标签
least_n_labels = label_counts.nsmallest(n).index


# 过滤出这些标签对应的数据
filtered_data_forecast = winddata[winddata['Label'].isin(least_n_labels)]

# 将风机容量也进行维度调整，便于运算
wind_capacity_reshaped = wind_capacity.reshape(-1, 24)

# 获取过滤数据的前24列数据（第25列为label），与对应的风机容量相乘，即为原始预测的风力发电，再转化为array形式，最终调整为1列的形式，便于保存
filtered_data_forecast = np.array(filtered_data_forecast.iloc[:, :24]*wind_capacity_reshaped[winddata['Label'].isin(least_n_labels)]).reshape(-1)

# 对真实风电数据进行维度调整，便于运算
wind_true_reshaped = wind_measured.reshape(-1, 24)

# 获取与过滤数据相对应（第n天对第n天）的真实风力发电数据，最终调整为1列的形式，便于保存
filtered_data_ture = wind_true_reshaped[winddata['Label'].isin(least_n_labels)].reshape(-1)

# 将过滤后的预测数据和真实数据均转化成dataframe形式
filtered_data = pd.DataFrame({'WindForecast': filtered_data_forecast, 'windTrue': filtered_data_ture, 'WindCapcity':wind_capacity_reshaped[winddata['Label'].isin(least_n_labels)].reshape(-1)})

# 输出结果
print(filtered_data)

# 保存为xlsx文件
filtered_data.to_excel(r'data\filtered_data.xlsx', index=False)


# 找出过滤后剩余的数据
filtered_data_left_forecast = winddata[~winddata['Label'].isin(least_n_labels)]

# 获取过滤后剩余的数据的前24列数据（第25列为label），与对应的风机容量相乘，即为原始预测的风力发电，再转化为array形式，最终调整为1列的形式，便于保存
filtered_data_left_forecast = np.array(filtered_data_left_forecast.iloc[:, :24]*wind_capacity_reshaped[~winddata['Label'].isin(least_n_labels)]).reshape(-1)

# 获取与过滤后剩余的数据相对应（第n天对第n天）的真实风力发电数据，最终调整为1列的形式，便于保存
filtered_data_left_ture = wind_true_reshaped[~winddata['Label'].isin(least_n_labels)].reshape(-1)

# 将过滤后的预测数据和真实数据均转化成dataframe形式
filtered_data_left = pd.DataFrame({'WindForecast': filtered_data_left_forecast, 'windTrue': filtered_data_left_ture, 'WindCapcity':wind_capacity_reshaped[~winddata['Label'].isin(least_n_labels)].reshape(-1)})

# 输出结果
print(filtered_data_left)

# 保存为xlsx文件
filtered_data_left.to_excel(r'data\filtered_data_left.xlsx', index=False)
'''

# ## 4. 风电误差数据保存

# # 将日前预测风电误差数据归一化
# wind_dayahead_forecast_error_Normalization = (wind_dayahead_forecast-wind_measured)/wind_capacity

# # 将风电数据改成一天一行的形式，便于DTC训练
# reshaped_error_data = wind_dayahead_forecast_error_Normalization.reshape(-1, 24)

# # 创建新的DataFrame
# new_df = pd.DataFrame(reshaped_error_data)

# # 写入到新的Excel文件（用于DTC训练）
# new_df.to_excel(r'C:\\Users\\余志航\\Desktop\\RBC和电力调度\\代码\\Deep-temporal-clustering\\windforecast_error.xlsx', index=False, header=False)

# ## 5. DTC训练完成后数据处理（风电误差分类）

# # 获取带有标签的数据
# winddata_error = pd.read_csv(r"data\results_data_error.csv")

# # 统计每种标签的数量
# label_counts = winddata_error['Label'].value_counts()

# # 找出数量最少的 n 种标签
# n = 10  # 找出数量最少的 n 种标签
# least_n_labels = label_counts.nsmallest(n).index


# # 过滤出这些标签对应的数据
# filtered_data_winderror = winddata_error[winddata_error['Label'].isin(least_n_labels)]

# # 将风机容量也进行维度调整，便于运算
# wind_capacity_reshaped = wind_capacity.reshape(-1, 24)

# # 获取过滤数据的前24列数据（第25列为label），与对应的风机容量相乘，即为原始预测的风力误差，再转化为array形式，最终调整为1列的形式，便于保存
# filtered_data_winderror = np.array(filtered_data_winderror.iloc[:, :24]*wind_capacity_reshaped[winddata_error['Label'].isin(least_n_labels)]).reshape(-1)

# # 对真实风电数据进行维度调整，便于运算
# wind_true_reshaped = wind_measured.reshape(-1, 24)

# # 获取与过滤数据相对应（第n天对第n天）的真实风力发电数据，最终调整为1列的形式，便于保存
# filtered_data_ture = wind_true_reshaped[winddata_error['Label'].isin(least_n_labels)].reshape(-1)

# # 对预测风电数据进行维度调整，便于运算
# wind_forecast_reshaped = wind_dayahead_forecast.reshape(-1, 24)

# # 获取与过滤数据相对应（第n天对第n天）的预测风力发电数据，最终调整为1列的形式，便于保存
# filtered_data_forecast = wind_forecast_reshaped[winddata_error['Label'].isin(least_n_labels)].reshape(-1)

# # 将过滤后的预测数据和真实数据均转化成dataframe形式
# filtered_data_error = pd.DataFrame({'Winderror': filtered_data_winderror,'WindForecast':filtered_data_forecast, 'windTrue': filtered_data_ture, 'WindCapcity':wind_capacity_reshaped[winddata_error['Label'].isin(least_n_labels)].reshape(-1)})

# # 输出结果
# print(filtered_data_error)

# # 保存为xlsx文件
# filtered_data_error.to_excel(r'data\filtered_data_error.xlsx', index=False)


# # 找出过滤后剩余的数据
# filtered_data_left_winderror = winddata_error[~winddata_error['Label'].isin(least_n_labels)]

# # 获取过滤后剩余的数据的前24列数据（第25列为label），与对应的风机容量相乘，即为原始预测的风力发电误差，再转化为array形式，最终调整为1列的形式，便于保存
# filtered_data_left_winderror = np.array(filtered_data_left_winderror.iloc[:, :24]*wind_capacity_reshaped[~winddata_error['Label'].isin(least_n_labels)]).reshape(-1)

# # 获取与过滤后剩余的数据相对应（第n天对第n天）的真实风力发电数据，最终调整为1列的形式，便于保存
# filtered_data_left_ture = wind_true_reshaped[~winddata_error['Label'].isin(least_n_labels)].reshape(-1)

# # 获取与过滤后剩余的数据相对应（第n天对第n天）的预测风力发电数据，最终调整为1列的形式，便于保存
# filtered_data_left_forecast = wind_forecast_reshaped[~winddata_error['Label'].isin(least_n_labels)].reshape(-1)

# # 将过滤后的预测数据和真实数据均转化成dataframe形式
# filtered_data_error_left = pd.DataFrame({'Winderror': filtered_data_left_winderror,'WindForecast':filtered_data_left_forecast, 'windTrue': filtered_data_left_ture, 'WindCapcity':wind_capacity_reshaped[~winddata_error['Label'].isin(least_n_labels)].reshape(-1)})

# # 输出结果
# print(filtered_data_error_left)

# # 保存为xlsx文件
# filtered_data_error_left.to_excel(r'data\filtered_data_error_left.xlsx', index=False)



## 6. Kmeans训练完成后数据处理（风电误差分类）

# 获取带有标签的数据
winddata_error = pd.read_csv(r"data\results_data_error.csv")

# 统计每种标签的数量
label_counts = winddata_error['Label'].value_counts()

# 创建 ExcelWriter 对象
# with pd.ExcelWriter(r'data\filtered_data_error_kmeans.xlsx', engine='xlsxwriter') as writer:
with pd.ExcelWriter(r'data\filtered_data_error.xlsx', engine='xlsxwriter') as writer:
    for n in range(len(label_counts)-1):
        # 找出数量最少的 n 种标签
        least_n_labels = label_counts.nsmallest(n + 1).index  # n + 1 因为 n 是从 0 开始的

        # 过滤出这些标签对应的数据
        filtered_data_winderror = winddata_error[winddata_error['Label'].isin(least_n_labels)]

        # 将风机容量进行维度调整
        wind_capacity_reshaped = wind_capacity.reshape(-1, 24)

        # 计算风力误差
        filtered_data_winderror = np.array(filtered_data_winderror.iloc[:, :24] * 
                                             wind_capacity_reshaped[winddata_error['Label'].isin(least_n_labels)]).reshape(-1)

        # 真实风电数据调整
        wind_true_reshaped = wind_measured.reshape(-1, 24)
        filtered_data_true = wind_true_reshaped[winddata_error['Label'].isin(least_n_labels)].reshape(-1)

        # 预测风电数据调整
        wind_forecast_reshaped = wind_dayahead_forecast.reshape(-1, 24)
        filtered_data_forecast = wind_forecast_reshaped[winddata_error['Label'].isin(least_n_labels)].reshape(-1)

        # 创建 DataFrame
        filtered_data_error = pd.DataFrame({
            'Winderror': filtered_data_winderror,
            'WindForecast': filtered_data_forecast,
            'windTrue': filtered_data_true,
            'WindCapacity': wind_capacity_reshaped[winddata_error['Label'].isin(least_n_labels)].reshape(-1)
        })

        # 输出结果
        print(filtered_data_error)

        # 保存到不同的工作表，命名为 "category n"
        sheet_name = f'categories {n + 1}'
        filtered_data_error.to_excel(writer, sheet_name=sheet_name, index=False)


# with pd.ExcelWriter(r'data\filtered_data_error_left_kmeans.xlsx', engine='xlsxwriter') as writer:
with pd.ExcelWriter(r'data\filtered_data_error_left.xlsx', engine='xlsxwriter') as writer:
    for n in range(len(label_counts)-1):
        # 找出数量最少的 n 种标签
        least_n_labels = label_counts.nsmallest(n + 1).index  # n + 1 因为 n 是从 0 开始的

        # 找出过滤后剩余的数据
        filtered_data_left_winderror = winddata_error[~winddata_error['Label'].isin(least_n_labels)]

        # 获取过滤后剩余的数据的前24列数据（第25列为label），与对应的风机容量相乘，即为原始预测的风力发电误差，再转化为array形式，最终调整为1列的形式，便于保存
        filtered_data_left_winderror = np.array(filtered_data_left_winderror.iloc[:, :24]*wind_capacity_reshaped[~winddata_error['Label'].isin(least_n_labels)]).reshape(-1)

        # 获取与过滤后剩余的数据相对应（第n天对第n天）的真实风力发电数据，最终调整为1列的形式，便于保存
        filtered_data_left_ture = wind_true_reshaped[~winddata_error['Label'].isin(least_n_labels)].reshape(-1)

        # 获取与过滤后剩余的数据相对应（第n天对第n天）的预测风力发电数据，最终调整为1列的形式，便于保存
        filtered_data_left_forecast = wind_forecast_reshaped[~winddata_error['Label'].isin(least_n_labels)].reshape(-1)

        # 将过滤后的预测数据和真实数据均转化成dataframe形式
        filtered_data_error_left = pd.DataFrame({
            'Winderror': filtered_data_left_winderror,
            'WindForecast':filtered_data_left_forecast,
            'windTrue': filtered_data_left_ture,
            'WindCapacity':wind_capacity_reshaped[~winddata_error['Label'].isin(least_n_labels)].reshape(-1)
        })

        # 输出结果
        print(filtered_data_error_left)

        # 保存到不同的工作表，命名为 "category n"
        sheet_name = f'categories {len(label_counts)-n-1}'
        filtered_data_error_left.to_excel(writer, sheet_name=sheet_name, index=False)