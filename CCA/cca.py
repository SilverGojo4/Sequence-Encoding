import pandas as pd

# 氨基酸對應物化性質的 Dataframe
eigenvector = pd.read_csv(filepath_or_buffer="eigenvectors.csv")

# 函數: Continuous Coding of Amino Acids
def cca(sequence, eigenvector):
    """
    參數: 
    - sequence: 字符串，待編碼的氨基酸序列
    - eigenvector: DataFrame，包含氨基酸的物化性質向量

    返回:
    - encoded_sequence: 列表，包含序列中每個氨基酸的編碼向量
    """

    # 建立一個字典對應各自氨基酸的編碼
    encoding_dict = eigenvector.set_index('Amino Acid').T.to_dict('list')

    # 初始化編碼後的序列空列表
    encoded_sequence = []

    for char in sequence:

        # 錯誤檢查 (沒出現過的字母)
        if char not in encoding_dict:
            raise ValueError("Character {} not in sequence base.".format(char))
        
        # 添加編碼到列表
        if char in encoding_dict:
            encoded_sequence.append(encoding_dict[char])

    return encoded_sequence