import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 10, 1 # mean and standard deviation
# np.random.nomral 함수를 이용해서 평균 0, 표준편차 0.1인 sample들을 1000개 추출한다.
s = np.random.normal(mu, sigma, 1000)

# sample들의 historgram을 출력한다.
count, bins, ignored = plt.hist(s, 30, normed=True)
# sample들을 이용해서 Gaussian Distribution의 shape을 재구축해서 line으로 그린다.
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
		np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()