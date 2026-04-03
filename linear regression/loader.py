import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ChengduHouseDataset(Dataset):
   
    
    def __init__(self, data_path, is_train=True, feature_columns=None):
   
        self.data_path = data_path
        self.is_train = is_train
        self.feature_columns = feature_columns
        self.load_and_preprocess()
    
    def load_and_preprocess(self):
    
        # 读取数据
        if isinstance(self.data_path, pd.DataFrame):
            df = self.data_path.copy()
        else:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据文件 {self.data_path} 不存在")
            df = pd.read_csv(self.data_path)
        
        # 保存原始数据框供后续使用
        self.df = df.copy()
        
        # 数据预处理
        df = self.preprocess_data(df)
        
        # 提取特征和标签
        if '单价（元/平方米）' in df.columns:
            # 移除ID列和目标列作为特征
            features_df = df.drop(['单价（元/平方米）'], axis=1, errors='ignore')
            # 确保所有特征都是数值类型
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            # 填充任何NaN值
            features_df = features_df.fillna(0)
            
            # 如果是训练模式，保存特征列名
            if self.is_train:
                self.feature_columns = features_df.columns.tolist()
            # 如果是测试模式，确保特征列与训练时一致
            else:
                if self.feature_columns:
                    # 添加缺失的列并设为0
                    for col in self.feature_columns:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    # 按训练时的列顺序排序
                    features_df = features_df[self.feature_columns]
            
            # 转换为numpy数组
            self.features = features_df.values.astype(np.float32)
            # 确保标签是浮点类型
            self.labels = df['单价（元/平方米）'].values.astype(np.float32)
        else:
            # 测试集没有标签
            features_df = df.copy()
            # 确保所有特征都是数值类型
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            # 填充任何NaN值
            features_df = features_df.fillna(0)
            
            # 如果是测试模式，确保特征列与训练时一致
            if not self.is_train and self.feature_columns:
                # 添加缺失的列并设为0
                for col in self.feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0
                # 按训练时的列顺序排序
                features_df = features_df[self.feature_columns]
            
            # 转换为numpy数组
            self.features = features_df.values.astype(np.float32)
            self.labels = None
    
    def preprocess_data(self, df):
    
        # 复制数据框
        df = df.copy()
        
        # 统一时间格式处理
        df = self.unify_time_format(df)
        
        # 处理缺失值
        df = self.handle_missing_values(df)
        
        # 异常值处理
        numerical_cols = ['建筑面积（平方米）']
        for col in numerical_cols:
            if col in df.columns:
                df = self.handle_outliers(df, col)
        
        # 特征工程
        df = self.feature_engineering(df)
        
        return df
    
    def unify_time_format(self, df):
        if '挂牌时间' not in df.columns:
            return df
    
        # 先处理字符串格式日期（如 "2021/12/3"）
        df['挂牌时间'] = pd.to_datetime(
            df['挂牌时间'],
            format='%Y/%m/%d',
            errors='coerce'  # 解析失败转为 NaT
        )
    
        # 处理 Excel 数字日期格式（如 "44248"）
        # 先提取还未解析成功的数字字符串，转为天数
        mask_excel = df['挂牌时间'].isna() & df['挂牌时间'].astype(str).str.match(r'^\d+(\.\d+)?$')     
        if mask_excel.any():
            days = pd.to_numeric(df.loc[mask_excel, '挂牌时间'], errors='coerce').astype(int)
            # Excel 日期基准是 1899-12-30，直接用 origin 参数更直观
        df.loc[mask_excel, '挂牌时间'] = pd.to_datetime(days, unit='D', origin='1899-12-30')
    
        # 转换为距离基准日期的天数特征
        base_date = pd.to_datetime('2020-01-01')
        df['挂牌时间'] = (df['挂牌时间'] - base_date).dt.days   
    
        # 缺失值填充（有效日期>0则用均值，否则填0）
        valid_days = df['挂牌时间'].dropna()
        if len(valid_days) > 0:
            df['挂牌时间'] = df['挂牌时间'].fillna(valid_days.mean())
        else:
            df['挂牌时间'] = 0
    
        return df   
    def handle_missing_values(self, df):
   
        # 数值型特征用均值填充
        numerical_cols = ['建筑面积（平方米）', '挂牌时间']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # 类别型特征用众数填充
        categorical_cols = ['房屋户型', '所在楼层', '户型结构', '建筑类型', 
                          '房屋朝向', '建筑结构', '装修情况', '配备电梯', 
                          '交易权属', '房屋用途']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '未知')
        
        return df
    
    def handle_outliers(self, df, col):
    
        # 确保是数值类型
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df[col].count() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 用中位数替换异常值
            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), 
                             df[col].median(), df[col])
        
        return df
    
    def feature_engineering(self, df):
 
        # 标准化数值特征
        numerical_cols = ['建筑面积（平方米）', '挂牌时间']
        for col in numerical_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
        
        # 类别特征One-Hot编码
        categorical_cols = ['房屋户型', '所在楼层', '户型结构', '建筑类型', 
                          '房屋朝向', '建筑结构', '装修情况', '配备电梯', 
                          '交易权属', '房屋用途']
        
        for col in categorical_cols:
            if col in df.columns:
                # 创建One-Hot编码
                one_hot = pd.get_dummies(df[col], prefix=col)
                # 合并到原数据框
                df = pd.concat([df, one_hot], axis=1)
                # 删除原始列
                df.drop(col, axis=1, inplace=True)
        
        return df
    
    def get_feature_importance(self, model):
  
        if hasattr(model, 'linear') and hasattr(model.linear, 'weight'):
            # 对于线性模型，权重的绝对值可以作为特征重要性
            weight = model.linear.weight.detach()
            importance = torch.abs(weight).numpy().flatten()
            feature_importance = dict(zip(self.feature_columns, importance))
            # 按重要性排序
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return sorted_importance
        return []
    
    def get_feature_selected_data(self, k=None):#k: 选择前k个重要特征
        
        if k is None or k >= len(self.feature_columns):
            return self.features
        
        # 基于特征重要性选择特征
        # 这里我们选择数值特征和部分类别特征
        numerical_cols = ['建筑面积（平方米）']
        selected_cols = []
        
        # 首先选择数值特征
        for col in numerical_cols:
            if col in self.feature_columns:
                selected_cols.append(col)
        
        # 然后选择部分类别特征（基于名称中包含的关键字）
        important_categories = ['装修情况', '配备电梯', '建筑结构', '所在楼层']
        for col in self.feature_columns:
            if any(cat in col for cat in important_categories):
                selected_cols.append(col)
        
        # 限制数量
        selected_cols = selected_cols[:k]
        
        # 获取对应的特征索引
        selected_indices = [self.feature_columns.index(col) for col in selected_cols if col in self.feature_columns]
        
        return self.features[:, selected_indices]
    
    def __len__(self):
  
        return len(self.features)
    
    def __getitem__(self, idx):

        if self.labels is not None:
            return {
                'features': torch.tensor(self.features[idx], dtype=torch.float32),
                'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
            }
        else:
            # 测试集没有标签
            return {
                'features': torch.tensor(self.features[idx], dtype=torch.float32)
            }

# 数据预处理函数
def process_chengdu_data(data_path):

    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整的文件路径
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 数据预处理
    dataset = ChengduHouseDataset(df)
    
    return dataset.df

