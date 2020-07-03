import numpy as np
import matplotlib.pyplot as plt
# from random import randint
from matplotlib.colors import LogNorm
from scipy.optimize import line_search as linsrch


# {{{ global variables
a, b = 2.4, 68.8
# }}} global vars


# {{{ class rosenbrock():
class rosenbrock():
    """class for Rosenbrock functions

    """
    # {{{ def func_rosen(self):
    def func_rosen(self):
        x, y = self.beta[0], self.beta[1]
        t1 = a - x
        t2 = y - x*x
        self.f = t1*t1 + b*t2*t2
        return self.f
    # }}} func_rosen(self)

    # {{{ def resid_rosen(beta):
    def resid_rosen(self):
        x, y = self.beta[0], self.beta[1]
        dfx = -2*(a - x) - 4*b*x*(y - x**2)
        dfy = 2*b*(y - x**2)
        self.r = np.array([dfx, dfy])
        return self.r
    # }}} resid_rosen(beta)

    # {{{ def obj_rosen(beta):
    def obj_rosen(self, *args):
        x, y = self.beta[0], self.beta[1]
        dfx = -2*(a - x) - 4*b*x*(y - x**2)
        dfy = 2*b*(y - x**2)
        return dfx**2 + dfy**2
    # }}} obj_rosen(beta)

    # {{{ def hess_rosen(beta):
    def hess_rosen(self):
        x, y = self.beta[0], self.beta[1]
        hess = np.zeros((2, 2), dtype=np.float)
        hess[0, 0] = 2 - 4*b*y + 12*b*x*x
        hess[1, 1] = 2*b
        hess[1, 0] = -4*b*x
        hess[0, 1] = hess[1, 0]
        self.hr = hess
        return self.hr
    # }}} hess_rosen(beta)

    # {{{ def grad_rosen(beta):
    def grad_S(self, *args):
        res = self.r
        hess = self.hr
        gradx = 2*hess[:, 0]@res
        grady = 2*hess[:, 1]@res
        self.g = np.array([gradx, grady])
        return self.g
    # }}} grad_rosen(beta)

    # {{{ def hessian(beta):
    def hessian(self):
        hess = np.zeros((2, 2), dtype=np.float)
        h_rosen = self.hr
        hess[0, 0] = 2*(h_rosen[0, 0]**2 +
                        h_rosen[0, 1]**2)
        hess[0, 1] = 2*(h_rosen[0, 0]*h_rosen[0, 1] +
                        h_rosen[1, 0]*h_rosen[1, 1])
        hess[1, 1] = 2*(h_rosen[1, 1]**2 +
                        h_rosen[1, 0]**2)
        hess[1, 0] = hess[0, 1]
        self.H = hess
        return self.H
    # }}} hessian(beta)

    # {{{ def __init__(self, beta):
    def __init__(self, beta):
        self.beta = beta
        self.f = self.func_rosen()
        self.r = self.resid_rosen()
        self.hr = self.hess_rosen()
        self.g = self.grad_S()
        self.H = self.hessian()
        """
        print(f" beta = {self.beta}\n" +
              f" f = {self.f}\n" +
              f" r = {self.r}\n" +
              f" g = {self.g}\n" +
              f" S = {((self.r)**2).sum()}\n" +
              f"=====================================")
        """
    # }}} __init__(self, beta)

    # {{{ def gauss_newton(rosen)
    def gauss_newton(self, damping):
        h = self.H
        g = self.g
        hi = np.linalg.pinv(h, rcond=1e-15)
        beta_corr = damping * hi@g.transpose()
        rosen_temp = rosen
        dpg, fc, gc, nf, of, ns = linsrch(rosen_temp.obj_rosen,
                                          rosen_temp.grad_S,
                                          rosen_temp.beta, beta_corr,
                                          c1=1e-4, c2=0.9)
        print(f" dpg = {dpg}, beta_corr = {beta_corr}")
        print(f" func evals = {fc}; grad evals = {gc}")
        self.beta = self.beta - dpg*beta_corr.reshape(2)
        return self.beta
    # }}} gauss_newton(J, res, p1, damping)

    # {{{ def iter_GN_rosen(self, damping, Niter):
    def iter_GN_rosen(self, damping, Niter, reslimit):
        beta_list = np.zeros((Niter+1, 2), dtype=np.float)
        res_list = np.zeros(Niter, dtype=np.float)
        beta_list[0, :] = self.beta.copy()
        self.count = 1
        for i in range(Niter):
            beta = self.gauss_newton(damping)
            self.__init__(beta)
            beta_list[i+1, :] = beta.copy()
            self.count += 1
            S = np.sqrt(((self.r)**2).sum())
            res_list[i] = S
            if S < reslimit:
                self.beta_list = beta_list
                self.S_list = res_list
                return None
        self.beta_list = beta_list
        self.S_list = res_list
        return None
    # }}} iter_GN_rosen(self, damping, Niter)

# }}} rosenbrock()


# {{{ def rosenbrock(a, b, N):
def gen_rosenbrock(a, b, beta_list, N):
    xmin = beta_list[:, 0].min()
    xmax = beta_list[:, 0].max()
    ymin = beta_list[:, 1].min()
    ymax = beta_list[:, 1].max()
    _xcent = a
    _ycent = a**2
    if (xmin < _xcent) and (xmax > _xcent):
        x = np.linspace(xmin - 5, xmax + 5, N).reshape(N, 1)
    else:
        x = np.linspace(_xcent - 5, _xcent + 5, N).reshape(N, 1)

    if (ymin < _ycent) and (ymax > _ycent):
        y = np.linspace(ymin - 5, ymax + 5, N).reshape(1, N)
    else:
        y = np.linspace(_ycent - 5, _ycent + 5, N).reshape(1, N)
    f = (a - x)**2 + b*(y - x*x)**2  # + (a - y)**2 + b*(x - y*y)**2
    Y, X = np.meshgrid(y, x)
    return X, Y, f
# }}} rosenbrock(a, b, N)


if __name__ == "__main__":
    plot_list = True
    plot_contour = True

    distance = 30
    beta = np.array([a, a**2]) + (distance - 2*distance*np.random.rand(2))
    beta = np.array([a+1, a**2+2])
    rosen = rosenbrock(beta)
    rosen.iter_GN_rosen(5e-1, 500, 1e-10)
    print(f" Iteration count = {rosen.count}")
    print(f" xmin = {rosen.beta[0]}; error = {(rosen.beta[0]-a)/a*100:5.2e}%")
    print(f" ymin = {rosen.beta[1]}; " +
          f"error = {(rosen.beta[1]-a**2)/a**2*100:5.2e}%")

    # plotting
    if plot_list:
        fig = plt.figure()
        if plot_contour:
            N_grid = 500
            x_rosen, y_rosen, f_rosen = gen_rosenbrock(a, b, rosen.beta_list,
                                                       N_grid)
            level_num = 20
            levels = np.linspace(np.log(f_rosen.min()/100),
                                 np.log(f_rosen.max()), level_num)
            im = plt.contourf(x_rosen, y_rosen, f_rosen,
                              levels=np.exp(levels), cmap=plt.cm.jet,
                              norm=LogNorm())
            plt.colorbar(im)
        plt.plot(rosen.beta_list[:rosen.count, 0],
                 rosen.beta_list[:rosen.count, 1], "-o")
        plt.plot(a, a**2, 'x', color='black')
        plt.show(fig)
        plt.close(fig)
        '''
        plt.figure()
        plt.semilogy(rosen.S_list[:rosen.count])
        plt.show()
        '''
