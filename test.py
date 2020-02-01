'''
@Author: greatpie
@Date: 2020-01-15 17:52:10
@LastEditTime : 2020-02-01 20:44:53
@LastEditors  : Please set LastEditors
@Description: For Xiaolu
@FilePath: /CuteXiao/entropy.py
'''


class EmtropyMethod:
    def __init__(self, index, positive, negative, row_name):
        if len(index) != len(row_name):
            raise Exception('数据指标行数与行名称数不符')
        if sorted(index.columns) != sorted(positive + negative):
            raise Exception('正项指标加负向指标不等于数据指标的条目数')

        self.index = index.copy().astype('float64')
        self.positive = positive
        self.negative = negative
        self.row_name = row_name

    def uniform(self):
        uniform_mat = self.index.copy()
        min_index = {
            column: min(uniform_mat[column])
            for column in uniform_mat.columns
        }
        max_index = {
            column: max(uniform_mat[column])
            for column in uniform_mat.columns
        }
        for i in range(len(uniform_mat)):
            for column in uniform_mat.columns:
                if column in self.negative:
                    uniform_mat[column][i] = (
                        uniform_mat[column][i] - min_index[column]) / (
                            max_index[column] - min_index[column])
                else:
                    uniform_mat[column][i] = (
                        max_index[column] - uniform_mat[column][i]) / (
                            max_index[column] - min_index[column])

        self.uniform_mat = uniform_mat
        return self.uniform_mat

    def calc_probability(self):
        try:
            p_mat = self.uniform_mat.copy()
        except AttributeError:
            raise Exception('你还没进行归一化处理，请先调用uniform方法')
        for column in p_mat.columns:
            sigma_x_1_n_j = sum(p_mat[column])
            p_mat[column] = p_mat[column].apply(
                lambda x_i_j: x_i_j / sigma_x_1_n_j
                if x_i_j / sigma_x_1_n_j != 0 else 1e-6)

        self.p_mat = p_mat
        return p_mat

    def calc_emtropy(self):
        try:
            self.p_mat.head(0)
        except AttributeError:
            raise Exception('你还没计算比重，请先调用calc_probability方法')

        import numpy as np
        e_j = -(1 / np.log(len(self.p_mat) + 1)) * np.array([
            sum([pij * np.log(pij) for pij in self.p_mat[column]])
            for column in self.p_mat.columns
        ])
        ejs = pd.Series(e_j, index=self.p_mat.columns, name='指标的熵值')

        self.emtropy_series = ejs
        return self.emtropy_series

    def calc_emtropy_redundancy(self):
        try:
            self.d_series = 1 - self.emtropy_series
            self.d_series.name = '信息熵冗余度'
        except AttributeError:
            raise Exception('你还没计算信息熵，请先调用calc_emtropy方法')

        return self.d_series

    def calc_weight(self):
        self.uniform()
        self.calc_probability()
        self.calc_emtropy()
        self.calc_emtropy_redundancy()
        self.weight = self.d_series / sum(self.d_series)
        self.weight.name = '权值'
        return self.weight

    def calc_score(self):
        self.calc_weight()

        import numpy as np
        self.score = pd.Series([
            np.dot(
                np.array(self.index[row:row + 1])[0], np.array(self.weight))
            for row in range(len(self.index))
        ],
                               index=self.row_name,
                               name='得分').sort_values(ascending=False)
        return self.score


if __name__ == '__main__':

    import pandas as pd
    df = pd.read_csv('灰色.csv')
    df = df.dropna().reset_index(drop=True)

    indexs = ['污水排放强度X1', '废气排放强度X2', '化学需氧量X3', '氨氮X4', '二氧化硫X5', '氮氧化物X6']
    Positive = indexs
    Negative = []

    grouper = df.groupby('year')

    weightDf = pd.DataFrame()
    scoreDf = pd.DataFrame()

    for name, group in grouper:
        groupeDf = group.reset_index()
        province = groupeDf['region']
        index = groupeDf[indexs]
        em = EmtropyMethod(index, Negative, Positive, province)
        em.calc_score()

        weight = em.weight.rename(name)
        score = em.score.rename(name)

        weightDf = weightDf.append(weight)
        scoreDf = scoreDf.append(score)

    weightDf.to_csv('weight.csv')
    scoreDf.T.to_csv('score.csv')