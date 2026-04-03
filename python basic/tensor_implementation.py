class Tensor:
 

    def __init__(self, data):
 
        # 验证输入是否为二维列表
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise ValueError("Only 2D list data is supported")
        
        # 验证所有行长度相同
        if not data:
            raise ValueError("Empty tensor not supported")
        
        row_length = len(data[0])
        for row in data:
            if len(row) != row_length:
                raise ValueError("All rows must have the same length")
        
        # 存储为纯Python列表
        self.data = [row.copy() for row in data]
    
    def shape(self):
        return (len(self.data), len(self.data[0]) if self.data else 0)#  返回tuple: 张量的形状，格式为(行数, 列数)
    
    def transpose(self):

        rows, cols = self.shape()
        transposed = [[self.data[i][j] for i in range(rows)] for j in range(cols)]
        return Tensor(transposed)
    
    def add(self, other):
    
        if isinstance(other, Tensor):
            if self.shape() != other.shape():
                raise ValueError("Tensors must have the same shape")# other是tensor的类
            result = []
            for i in range(len(self.data)):
                row = []
                for j in range(len(self.data[0])):
                    row.append(self.data[i][j] + other.data[i][j])
                result.append(row)
            return Tensor(result)
        elif isinstance(other, (int, float)):# other是标量
            result = []
            for i in range(len(self.data)):
                row = []
                for j in range(len(self.data[0])):
                    row.append(self.data[i][j] + other)
                result.append(row)
            return Tensor(result)
        else:
            raise TypeError("Unsupported type for addition")
    
    def sub(self, other):

        if isinstance(other, Tensor):
            if self.shape() != other.shape():
                raise ValueError("Tensors must have the same shape")
            result = []
            for i in range(len(self.data)):
                row = []
                for j in range(len(self.data[0])):
                    row.append(self.data[i][j] - other.data[i][j])
                result.append(row)
            return Tensor(result)
        elif isinstance(other, (int, float)):
            result = []
            for i in range(len(self.data)):
                row = []
                for j in range(len(self.data[0])):
                    row.append(self.data[i][j] - other)
                result.append(row)
            return Tensor(result)
        else:
            raise TypeError("Unsupported type for subtraction")
    
    def mul(self, other):
       
        if isinstance(other, Tensor):
            if self.shape() != other.shape():
                raise ValueError("Tensors must have the same shape")
            result = []
            for i in range(len(self.data)):
                row = []
                for j in range(len(self.data[0])):
                    row.append(self.data[i][j] * other.data[i][j])
                result.append(row)
            return Tensor(result)
        elif isinstance(other, (int, float)):
            result = []
            for i in range(len(self.data)):
                row = []
                for j in range(len(self.data[0])):
                    row.append(self.data[i][j] * other)
                result.append(row)
            return Tensor(result)
        else:
            raise TypeError("Unsupported type for multiplication")
    
    def matmul(self, other):
        
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires another Tensor")
        
        rows_a, cols_a = self.shape()
        rows_b, cols_b = other.shape()
        
        if cols_a != rows_b:
            raise ValueError("Number of columns in first tensor must match number of rows in second tensor")
        
        # 矩阵乘法实现
        result = []
        for i in range(rows_a):
            row = []
            for j in range(cols_b):
                # 计算点积
                dot_product = 0
                for k in range(cols_a):
                    dot_product += self.data[i][k] * other.data[k][j]# 计算点积
                row.append(dot_product)
            result.append(row)
        
        return Tensor(result)
    
    def mean(self, axis):
        
        
        if axis == 0:  # 按列计算
            rows, cols = self.shape()
            means = []
            for j in range(cols):
                total = 0
                for i in range(rows):
                    total += self.data[i][j]
                means.append(total / rows)
            return means
        else:  # 按行计算
            means = []
            for row in self.data:
                total = sum(row)
                means.append(total / len(row))
            return means
    
    def std(self, axis):
        
        means = self.mean(axis)
        
        if axis == 0:  # 按列计算
            rows, cols = self.shape()
            stds = []
            for j in range(cols):
                variance = 0
                for i in range(rows):
                    variance += (self.data[i][j] - means[j]) ** 2
                stds.append((variance / rows) ** 0.5)
            return stds
        else:  # 按行计算
            stds = []
            for i, row in enumerate(self.data):
                variance = 0
                for val in row:
                    variance += (val - means[i]) ** 2
                stds.append((variance / len(row)) ** 0.5)
            return stds
    
    def min(self, axis):
        
        if axis == 0:  # 按列计算
            rows, cols = self.shape()
            mins = []
            for j in range(cols):
                min_val = self.data[0][j]
                for i in range(1, rows):
                    if self.data[i][j] < min_val:
                        min_val = self.data[i][j]
                mins.append(min_val)
            return mins
        else:  # 按行计算
            mins = []
            for row in self.data:
                mins.append(min(row))
            return mins
    
    def max(self, axis):
        
        if axis == 0:  # 按列计算
            rows, cols = self.shape()
            maxs = []
            for j in range(cols):
                max_val = self.data[0][j]
                for i in range(1, rows):
                    if self.data[i][j] > max_val:
                        max_val = self.data[i][j]
                maxs.append(max_val)
            return maxs
        else:  # 按行计算
            maxs = []
            for row in self.data:
                maxs.append(max(row))
            return maxs
    
    def standardize(self):
 
        rows, cols = self.shape()
        # 计算每列的均值和标准差
        means = self.mean(0)
        stds = self.std(0)
        
        # 避免除零错误
        for i in range(len(stds)):
            if stds[i] == 0:
                stds[i] = 1
        
        # 应用标准化
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append((self.data[i][j] - means[j]) / stds[j])
            result.append(row)
        
        return Tensor(result)
    
    # 运算符重载：让Tensor对象可以使用Python的运算符
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    def __getitem__(self, key):
       

        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_slice = key
            rows = self.data[row_slice]
            if isinstance(rows, list):
                result = []
                for row in rows:
                    result.append(row[col_slice])
                return Tensor(result)
        elif isinstance(key, int):
            return Tensor([self.data[key]])
        raise NotImplementedError("Only basic slicing is supported")
    
    def __str__(self):
  
        return str(self.data)
    
    @property
    def T(self):

        return self.transpose()

# Step 1: 创建张量实例
def step1():
    A = Tensor([[1, 2, 3], [4, 5, 6]])
    B = Tensor([[7, 8, 9], [10, 11, 12]])
    return A, B

# Step 2: 张量实例与运算
def step2():

    # 调用step1创建张量实例
    A, B = step1()
    
    print("Tensor A:")
    print(A)
    print("Shape of A:", A.shape())
    print("Transpose of A:")
    print(A.transpose())
    print("Mean of A by columns:", A.mean(0))
    print("Mean of A by rows:", A.mean(1))
    
    print("\nTensor B:")
    print(B)
    
    # 使用add方法进行加法运算
    print("\nA.add(B):")
    print(A.add(B))
    
    # 使用sub方法进行减法运算
    print("B.sub(A):")
    print(B.sub(A))
    
    # 使用matmul方法进行矩阵乘法
    try:
        print("\nA.matmul(B) (matrix multiplication):")
        print(A.matmul(B))
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\nA.matmul(B.transpose()):")
    C = A.matmul(B.transpose())
    print(C)
    print("Standardized C:")
    C_standardized = C.standardize()
    print(C_standardized)
    print("Min of C by columns:", C.min(0))
    print("Max of C by columns:", C.max(0))
    print("Min of C by rows:", C.min(1))
    print("Max of C by rows:", C.max(1))

# Step 3: 运算符重载示例
def step3():

    # 调用step1创建张量实例
    A, B = step1()
    
    print("Step 3: 运算符重载示例")
    print("C = A + B:")
    C = A + B
    print(C)
    
    print("D = B - A:")
    D = B - A
    print(D)
    
    print("E = A * 2:")
    E = A * 2
    print(E)
    
    print("F = A @ B.T:")
    F = A @ B.transpose()
    print(F)
    
    print("H = A[0:2, 1:3] * 3:")
    H = A[0:2, 1:3] * 3
    print(H)

# Extension 2: 实现二维Tensor的广播机制
def extension2():
    """
    从右往左比较两个张量的维度大小
    对于每个维度，要么大小相同，要么其中一个为1
    维度为1的张量会被扩展为与另一个张量相同的大小
    """
    # 实现广播加法
    def broadcast_add(self, other):
        if isinstance(other, list):
            # 检查other是否为一维列表且长度等于张量的列数
            if isinstance(other, list) and all(isinstance(x, (int, float)) for x in other):
                if len(other) == self.shape()[1]:
                    # 广播加法：行向量与矩阵的每一行相加
                    result = []
                    for row in self.data:
                        new_row = []
                        for i, val in enumerate(row):
                            new_row.append(val + other[i])# 行向量与矩阵每一行相加
                        result.append(new_row)
                    return Tensor(result)
        return self.add(other)
    
    # 测试广播加法
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    res1 = broadcast_add(t, [10, 20, 30])
    res2 = t * 2
    
    print("\n测试广播加法:")
    print("t =", t)
    print("res1 = t + [10, 20, 30] =", res1)
    print("res2 = t * 2 =", res2)

if __name__ == "__main__":

    print("Step 1: 创建张量实例")
    A, B = step1()
    print("张量A和B已创建\n")
    
    print("Step 2: 张量实例与运算")
    step2()
    
    print("\nStep 3: 运算符重载示例")
    step3()
    
    extension2()
