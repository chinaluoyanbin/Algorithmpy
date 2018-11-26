# -*- UTF-8 -*-
from scipy import stats
import numpy as np


class ExpectationMaximization(object):
    def __init__(self, tol=1e-6, iterations=10000, theta=[0, 0]):
        self.tol = tol
        self.iterations = iterations
        self.theta = theta

    def em_single_step(self, trainDatas, initValues):
        """
        EM算法单步迭代
        """
        expertation = {
            'A': {'P': 0, 'N': 0},
            'B': {'P': 0, 'N': 0}
        }
        A_initValue = initValues[0]
        B_initValue = initValues[1]

        # E step
        for trainData in trainDatas:
            num_TrainData = len(trainData)
            num_Positive = trainData.sum()
            num_Negative = num_TrainData - num_Positive
            A_Contribution = stats.binom.pmf(num_Positive, num_TrainData, A_initValue)
            B_Contribution = stats.binom.pmf(num_Positive, num_TrainData, B_initValue)
            A_Probability = A_Contribution / (A_Contribution + B_Contribution)
            B_Probability = B_Contribution / (A_Contribution + B_Contribution)
            # 更新当前参数下A、B的P、N的期望值
            expertation['A']['P'] = A_Probability * num_Positive
            expertation['A']['N'] = A_Probability * num_Negative
            expertation['B']['P'] = B_Probability * num_Positive
            expertation['B']['N'] = B_Probability * num_Negative
        
        # M step
        new_A_initValue = expertation['A']['P'] / (expertation['A']['P'] + expertation['A']['N'])
        new_B_initValue = expertation['B']['P'] / (expertation['B']['P'] + expertation['B']['N'])

        return [new_A_initValue, new_B_initValue]

    def fit(self, trainDatas, initValues):
        iteration = 0
        while iteration < self.iterations:
            new_initValues = self.em_single_step(trainDatas, initValues)
            tol = np.abs(new_initValues[0] - initValues[0])
            if tol < self.tol:
                break
            else:
                initValues = new_initValues
                iteration += 1

        self.theta = new_initValues


if __name__ == '__main__':
    # instances
    trainDatas = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                           [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                           [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

    # have a try
    em = ExpectationMaximization()
    em.fit(trainDatas=trainDatas, initValues=[0.1, 0.9])
    print(em.theta)
