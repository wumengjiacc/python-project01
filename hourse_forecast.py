import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import re
import warnings

warnings.filterwarnings('ignore')

# 内存优化配置
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'


# 生成模拟数据
def generate_sample_Data(num_samples=10000):
    """"""
    np.random.seed(42)
    locations = ['downtown', 'suburb', 'rural', 'beachfront', 'mountain', 'urban', 'quiet', 'busy']
    features = ['renovated', 'modern', 'classic', 'spacious', 'cozy', 'cozy', 'luxury', 'basic']
    conditions = ['excellent', 'good', 'fair', 'needs work']

    descriptions = []
    desc_adjustments = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        loc = np.random.choice(locations, 2, replace=False)
        feat = np.random.choice(features, 3, replace=False)
        cond = np.random.choice(conditions, 1)[0]
        desc = f"{' '.join(loc)} house with {' '.join(feat)} features in {cond} condition"
        descriptions.append(desc)
        adj = 0
        if 'luxury' in desc: adj += 50000
        if 'renovated' in desc: adj += 30000
        if 'needs work' in desc: adj -= 20000
        if 'downtown' in desc: adj += 40000
        if 'rural' in desc: adj -= 15000
        desc_adjustments[i] = adj

    # 生成数值特征
    data = pd.DataFrame({
        'description': descriptions,
        'bedrooms': np.random.randint(1, 6, num_samples),
        'bathrooms': np.random.randint(1, 4, num_samples),
        'sqft': np.random.randint(800, 4000, num_samples),
        'year_built': np.random.randint(1950, 2022, num_samples)
    })

    # 基于特征生成房价
    base_price = data['sqft'].values * 150
    bedroom_bonus = data['bedrooms'].values * 25000
    bathroom_bonus = data['bathrooms'].values * 15000
    age_factor = (2026 - data['year_built']).values * -500

    noise = np.random.normal(0, 10000, num_samples).astype(int)
    price = base_price + bedroom_bonus + bathroom_bonus + age_factor + desc_adjustments + noise
    price = np.clip(price, 100000, 5000000)
    data['price'] = price.astype(int)

    # 添加统计数据
    print(f"生成统计数据")
    print(f" 价格范围：${data['price'].min():,} - ${data['price'].max():,}")
    print(f" 平均价格: ${data['price'].mean():,.2f}")
    print(f" 价格标准差: ${data['price'].std():,.2f}")

    return data



if __name__ == '__main__':
    data = generate_sample_Data()
    # print(data)
    # print("形状：", data.shape)
    # print("类型:", data.dtypes)
    # print("基本信息", data.info())
    # print("统计描述", data.describe())
    print("最高价格")
    print(data.nlargest(5,'price'))

    print("最低价格")
    print(data.nsmallest(5, 'price'))

    print("前5行数据：")
    print(data.head(5))
    print("后5行数据：")
    print(data.tail(5))

    print("随机五行")
    print(data.sample(5))
