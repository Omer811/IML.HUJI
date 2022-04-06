from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.express as px
MAP_SIZE = 200

SAMPLE_SIZE = 1000

VAR = 1

MEAN = 10



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(MEAN, VAR, size=SAMPLE_SIZE)
    gauss = UnivariateGaussian ()
    gauss.fit(samples)
    print(f"({gauss.mu_},{gauss.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    results = np.empty((int(SAMPLE_SIZE/10),2), dtype=float)
    for size in range(10, SAMPLE_SIZE, 10):
        gauss.fit(samples[:size])
        results[int(size/10),0] = size
        results[int(size/10),1] = np.abs(gauss.mu_ - MEAN)

    fig = px.line( x=results[1:,0], y=results[1:,1], markers=True,labels={
        'x':"Sample Size",'y':"Distance from mean"})
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    samples.sort()
    pdf = gauss.pdf(samples)
    fig = px.line(x=samples, y=pdf, markers=True, labels={
        'x': "X", 'y': "PDF(X)"})
    fig.show()


def test_multivariate_gaussian():
    mean = np.array([0,0,4,0]).T
    cov = np.array([[1 ,0.2 ,0 ,0.5],[0.2 ,2 ,0 ,0],[0 ,0 ,1 ,0],[0.5 ,0 ,0
        ,1]])
    samples = np.random.multivariate_normal(mean,cov,size=SAMPLE_SIZE)

    # Question 4 - Draw samples and print fitted model
    gauss = MultivariateGaussian()
    gauss.fit(samples)
    print(gauss.mu_)
    print(gauss.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, MAP_SIZE)
    f3 = np.linspace(-10, 10, MAP_SIZE)
    res = np.empty((MAP_SIZE, MAP_SIZE), dtype=float)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            res[i,j] = MultivariateGaussian.\
                log_likelihood(np.array([f3[i],0,f1[j],0]).T,cov,samples)
    fig = px.imshow(res,x=f3,y=f1,labels={'x':"f3",'y':"f1",
                                          "color":"log-likelihood"})
    fig.show()
    # Question 6 - Maximum likelihood
    max_index = np.unravel_index(np.argmax(res),res.shape)
    print(f"f1: {np.around(f1[max_index[0]],4)} f3:"
          f"{np.around(f3[max_index[1]],4)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

