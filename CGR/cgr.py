import numpy as np
import math

# 函數: 根據選擇的序列類型決定序列編碼基底
def get_seq_base(seq_type):
    """
    參數: 
    - seq_type: 字符串，指定序列類型 ('AMINO', 'DNA', 'DIGITS', 'LETTERS')

    返回: 
    - 序列基底的列表，若類型無效則返回 None
    """

    # 定義序列基底
    bases = {
        'AMINO': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
        'DNA': ['A', 'G', 'T', 'C'],
        'DIGITS': [str(digit) for digit in range(10)],
        'LETTERS': [chr(i) for i in range(65, 91)]
    }

    return bases.get(seq_type.upper(), None)

# 函數: 取得 n 邊、半徑為 r 的正多邊形 (基於單位圓上) 的座標
def get_coordinate(n, r):
    """
    參數: 
    - n: 正多邊形的邊數
    - r: 正多邊形的半徑

    返回: 
    - x,y 座標
    """

    # 特殊情況 n=4 (正方形)，此時座標固定
    if n == 4:
        x = np.array([1, -1, -1, 1]) * r
        y = np.array([-1, -1, 1, 1]) * r
    else:
        angles = [(2*i + 1)/n * np.pi for i in range(n)]  # 列表推導式用於創建調整後的角度
        x = r * np.sin(angles)  # 計算 x 座標
        y = r * np.cos(angles)  # 計算 y 座標

    return x, y

# 函數: 基於混沌遊戲表示演算法計算 n 邊正多邊形的比例因子 (前進頂點的距離)
def calculate_scaling_factor(n):
    """
    參數: 
    - n: 正多邊形的邊數。

    返回: 
    - 正多邊形的比例因子
    """
    if n == 4:  # 對於正方形 (或 DNA 基底)，縮放因子為 0.5
        return 0.5
    else:
        # 根據邊數 n 計算縮放因子
        return 1 - (np.sin(np.pi / n) / (np.sin(np.pi / n) + np.sin(np.pi / n + 2 * np.pi * (np.floor(n / 4) / n))))
    
# 函數: Chaos Game Representation Encoding
def cgr(sequence, seq_base_type, res):
    """
    參數: 
    - sequence: 待編碼的序列
    - seq_base_type: 序列的類型
    - res: 頻率矩陣的解析度

    返回: 
    - 頻率矩陣的 numpy.ndarray
    """

    # 獲取序列編碼基底並錯誤檢查
    seq_base = get_seq_base(seq_base_type)
    if not seq_base:
        raise ValueError("Invalid seq_base_type provided.")

    # 取得 n 邊、半徑為 r 的正多邊形的座標
    n = len(seq_base)
    x_coords, y_coords = get_coordinate(n, 1)  # 為了簡化計算，假設單位圓

    # 計算前進距離
    sf = calculate_scaling_factor(n)

    # 創建了一個大小為 res x res 的矩陣，初始值為 0
    freq_matrix = np.zeros((res, res), dtype=int)

    # 設定中心點為 (0, 0)
    current_point = np.array([0, 0])

    # 將序列映射到頂點並更新頻率矩陣
    for char in sequence:

        # 錯誤檢查 (沒出現過的字母)
        if char not in seq_base:
            raise ValueError("Character {} not in sequence base.".format(char))
        
        # 找出對應索引
        index = seq_base.index(char)

        # 按前進距離向對應頂點移動
        target_vertex = np.array([x_coords[index], y_coords[index]]) # 計算目標頂點座標
        current_point = current_point + sf * (target_vertex-current_point) # 更新當前點的位置

        # 更新頻率矩陣
        # 將 current_point 轉換為矩陣索引
        x_matrix_idx = math.ceil(current_point[0] * res) - 1
        y_matrix_idx = math.ceil(current_point[1] * res) - 1
        freq_matrix[x_matrix_idx, y_matrix_idx] += 1

    return freq_matrix