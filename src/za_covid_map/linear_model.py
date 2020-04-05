# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# License: BSD 3 clause

import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
import scipy.optimize
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2
from scipy.special import expit, logit, xlogy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import deprecated

DistributionBoundary = namedtuple("DistributionBoundary",
                                  ("value", "inclusive"))


def _num_samples(X):
    return X.shape[0]
    
def _check_sample_weight(sample_weight, X, dtype=None):
    """Validate sample weights.
    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)
    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
       Input sample weights.
    X : nd-array, list or sparse matrix
        Input data.
    dtype: dtype
       dtype of the validated `sample_weight`.
       If None, and the input `sample_weight` is an array, the dtype of the
       input is preserved; otherwise an array with the default numpy dtype
       is be allocated.  If `dtype` is not one of `float32`, `float64`,
       `None`, the output will be of dtype `float64`.
    Returns
    -------
    sample_weight : ndarray, shape (n_samples,)
       Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
            order="C"
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight.shape == {}, expected {}!"
                             .format(sample_weight.shape, (n_samples,)))
    return sample_weight

class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is None:
        # line search failed: try different one.
        ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                 old_fval, old_old_fval, **kwargs)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def _cg(fhess_p, fgrad, maxiter, tol):
    """
    Solve iteratively the linear system 'fhess_p . xsupi = fgrad'
    with a conjugate gradient descent.

    Parameters
    ----------
    fhess_p : callable
        Function that takes the gradient as a parameter and returns the
        matrix product of the Hessian and gradient

    fgrad : ndarray, shape (n_features,) or (n_features + 1,)
        Gradient vector

    maxiter : int
        Number of CG iterations.

    tol : float
        Stopping criterion.

    Returns
    -------
    xsupi : ndarray, shape (n_features,) or (n_features + 1,)
        Estimated solution
    """
    xsupi = np.zeros(len(fgrad), dtype=fgrad.dtype)
    ri = fgrad
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)

    while i <= maxiter:
        if np.sum(np.abs(ri)) <= tol:
            break

        Ap = fhess_p(psupi)
        # check curvature
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= 3 * np.finfo(np.float64).eps:
            break
        elif curv < 0:
            if i > 0:
                break
            else:
                # fall back to steepest descent direction
                xsupi += dri0 / curv * psupi
                break
        alphai = dri0 / curv
        xsupi += alphai * psupi
        ri = ri + alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        i = i + 1
        dri0 = dri1          # update np.dot(ri,ri) for next time.

    return xsupi


@deprecated("newton_cg is deprecated in version "
            "0.22 and will be removed in version 0.24.")
def newton_cg(grad_hess, func, grad, x0, args=(), tol=1e-4,
              maxiter=100, maxinner=200, line_search=True, warn=True):
    return _newton_cg(grad_hess, func, grad, x0, args, tol, maxiter,
                      maxinner, line_search, warn)


def _newton_cg(grad_hess, func, grad, x0, args=(), tol=1e-4,
               maxiter=100, maxinner=200, line_search=True, warn=True):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Parameters
    ----------
    grad_hess : callable
        Should return the gradient and a callable returning the matvec product
        of the Hessian.

    func : callable
        Should return the value of the function.

    grad : callable
        Should return the function value and the gradient. This is used
        by the linesearch functions.

    x0 : array of float
        Initial guess.

    args : tuple, optional
        Arguments passed to func_grad_hess, func and grad.

    tol : float
        Stopping criterion. The iteration will stop when
        ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    maxiter : int
        Number of Newton iterations.

    maxinner : int
        Number of CG iterations.

    line_search : boolean
        Whether to use a line search or not.

    warn : boolean
        Whether to warn when didn't converge.

    Returns
    -------
    xk : ndarray of float
        Estimated minimum.
    """
    x0 = np.asarray(x0).flatten()
    xk = x0
    k = 0

    if line_search:
        old_fval = func(x0, *args)
        old_old_fval = None

    # Outer loop: our Newton iteration
    while k < maxiter:
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - fgrad f(xk) starting from 0.
        fgrad, fhess_p = grad_hess(xk, *args)

        absgrad = np.abs(fgrad)
        if np.max(absgrad) <= tol:
            break

        maggrad = np.sum(absgrad)
        eta = min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad

        # Inner loop: solve the Newton update by conjugate gradient, to
        # avoid inverting the Hessian
        xsupi = _cg(fhess_p, fgrad, maxiter=maxinner, tol=termcond)

        alphak = 1.0

        if line_search:
            try:
                alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                    _line_search_wolfe12(func, grad, xk, xsupi, fgrad,
                                         old_fval, old_old_fval, args=args)
            except _LineSearchError:
                warnings.warn('Line Search failed')
                break

        xk = xk + alphak * xsupi        # upcast if necessary
        k += 1

    if warn and k >= maxiter:
        warnings.warn("newton-cg failed to converge. Increase the "
                      "number of iterations.", ConvergenceWarning)
    return xk, k


def _check_optimize_result(solver, result, max_iter=None,
                           extra_warning_msg=None):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    solver: str
       solver name. Currently only `lbfgs` is supported.
    result: OptimizeResult
       result of the scipy.optimize.minimize function
    max_iter: {int, None}
       expected maximum number of iterations

    Returns
    -------
    n_iter: int
       number of iterations
    """
    # handle both scipy and scikit-learn solver names
    if solver == "lbfgs":
        if result.status != 0:
            warning_msg = (
                "{} failed to converge (status={}):\n{}.\n\n"
                "Increase the number of iterations (max_iter) "
                "or scale the data as shown in:\n"
                "    https://scikit-learn.org/stable/modules/"
                "preprocessing.html"
            ).format(solver, result.status, result.message.decode("latin1"))
            if extra_warning_msg is not None:
                warning_msg += "\n" + extra_warning_msg
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
        if max_iter is not None:
            # In scipy <= 1.0.0, nit may exceed maxiter for lbfgs.
            # See https://github.com/scipy/scipy/issues/7854
            n_iter_i = min(result.nit, max_iter)
        else:
            n_iter_i = result.nit
    else:
        raise NotImplementedError

    return n_iter_i
    
class ExponentialDispersionModel(metaclass=ABCMeta):
    r"""Base class for reproductive Exponential Dispersion Models (EDM).

    The pdf of :math:`Y\sim \mathrm{EDM}(y_\textrm{pred}, \phi)` is given by

    .. math:: p(y| \theta, \phi) = c(y, \phi)
        \exp\left(\frac{\theta y-A(\theta)}{\phi}\right)
        = \tilde{c}(y, \phi)
            \exp\left(-\frac{d(y, y_\textrm{pred})}{2\phi}\right)

    with mean :math:`\mathrm{E}[Y] = A'(\theta) = y_\textrm{pred}`,
    variance :math:`\mathrm{Var}[Y] = \phi \cdot v(y_\textrm{pred})`,
    unit variance :math:`v(y_\textrm{pred})` and
    unit deviance :math:`d(y,y_\textrm{pred})`.

    Methods
    -------
    deviance
    deviance_derivative
    in_y_range
    unit_deviance
    unit_deviance_derivative
    unit_variance

    References
    ----------
    https://en.wikipedia.org/wiki/Exponential_dispersion_model.
    """

    def in_y_range(self, y):
        """Returns ``True`` if y is in the valid range of Y~EDM.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.
        """
        # Note that currently supported distributions have +inf upper bound

        if not isinstance(self._lower_bound, DistributionBoundary):
            raise TypeError('_lower_bound attribute must be of type '
                            'DistributionBoundary')

        if self._lower_bound.inclusive:
            return np.greater_equal(y, self._lower_bound.value)
        else:
            return np.greater(y, self._lower_bound.value)

    @abstractmethod
    def unit_variance(self, y_pred):
        r"""Compute the unit variance function.

        The unit variance :math:`v(y_\textrm{pred})` determines the variance as
        a function of the mean :math:`y_\textrm{pred}` by
        :math:`\mathrm{Var}[Y_i] = \phi/s_i*v(y_\textrm{pred}_i)`.
        It can also be derived from the unit deviance
        :math:`d(y,y_\textrm{pred})` as

        .. math:: v(y_\textrm{pred}) = \frac{2}{
            \frac{\partial^2 d(y,y_\textrm{pred})}{
            \partialy_\textrm{pred}^2}}\big|_{y=y_\textrm{pred}}

        See also :func:`variance`.

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Predicted mean.
        """

    @abstractmethod
    def unit_deviance(self, y, y_pred, check_input=False):
        r"""Compute the unit deviance.

        The unit_deviance :math:`d(y,y_\textrm{pred})` can be defined by the
        log-likelihood as
        :math:`d(y,y_\textrm{pred}) = -2\phi\cdot
        \left(loglike(y,y_\textrm{pred},\phi) - loglike(y,y,\phi)\right).`

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.

        check_input : bool, default=False
            If True raise an exception on invalid y or y_pred values, otherwise
            they will be propagated as NaN.
        Returns
        -------
        deviance: array of shape (n_samples,)
            Computed deviance
        """

    def unit_deviance_derivative(self, y, y_pred):
        r"""Compute the derivative of the unit deviance w.r.t. y_pred.

        The derivative of the unit deviance is given by
        :math:`\frac{\partial}{\partialy_\textrm{pred}}d(y,y_\textrm{pred})
             = -2\frac{y-y_\textrm{pred}}{v(y_\textrm{pred})}`
        with unit variance :math:`v(y_\textrm{pred})`.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.
        """
        return -2 * (y - y_pred) / self.unit_variance(y_pred)

    def deviance(self, y, y_pred, weights=1):
        r"""Compute the deviance.

        The deviance is a weighted sum of the per sample unit deviances,
        :math:`D = \sum_i s_i \cdot d(y_i, y_\textrm{pred}_i)`
        with weights :math:`s_i` and unit deviance
        :math:`d(y,y_\textrm{pred})`.
        In terms of the log-likelihood it is :math:`D = -2\phi\cdot
        \left(loglike(y,y_\textrm{pred},\frac{phi}{s})
        - loglike(y,y,\frac{phi}{s})\right)`.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.

        weights : {int, array of shape (n_samples,)}, default=1
            Weights or exposure to which variance is inverse proportional.
        """
        return np.sum(weights * self.unit_deviance(y, y_pred))

    def deviance_derivative(self, y, y_pred, weights=1):
        r"""Compute the derivative of the deviance w.r.t. y_pred.

        It gives :math:`\frac{\partial}{\partial y_\textrm{pred}}
        D(y, \y_\textrm{pred}; weights)`.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        y_pred : array, shape (n_samples,)
            Predicted mean.

        weights : {int, array of shape (n_samples,)}, default=1
            Weights or exposure to which variance is inverse proportional.
        """
        return weights * self.unit_deviance_derivative(y, y_pred)


class TweedieDistribution(ExponentialDispersionModel):
    r"""A class for the Tweedie distribution.

    A Tweedie distribution with mean :math:`y_\textrm{pred}=\mathrm{E}[Y]`
    is uniquely defined by it's mean-variance relationship
    :math:`\mathrm{Var}[Y] \propto y_\textrm{pred}^power`.

    Special cases are:

    ===== ================
    Power Distribution
    ===== ================
    0     Normal
    1     Poisson
    (1,2) Compound Poisson
    2     Gamma
    3     Inverse Gaussian

    Parameters
    ----------
    power : float, default=0
            The variance power of the `unit_variance`
            :math:`v(y_\textrm{pred}) = y_\textrm{pred}^{power}`.
            For ``0<power<1``, no distribution exists.
    """
    def __init__(self, power=0):
        self.power = power

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        # We use a property with a setter, to update lower and
        # upper bound when the power parameter is updated e.g. in grid
        # search.
        if not isinstance(power, numbers.Real):
            raise TypeError('power must be a real number, input was {0}'
                            .format(power))

        if power <= 0:
            # Extreme Stable or Normal distribution
            self._lower_bound = DistributionBoundary(-np.Inf, inclusive=False)
        elif 0 < power < 1:
            raise ValueError('Tweedie distribution is only defined for '
                             'power<=0 and power>=1.')
        elif 1 <= power < 2:
            # Poisson or Compound Poisson distribution
            self._lower_bound = DistributionBoundary(0, inclusive=True)
        elif power >= 2:
            # Gamma, Positive Stable, Inverse Gaussian distributions
            self._lower_bound = DistributionBoundary(0, inclusive=False)
        else:  # pragma: no cover
            # this branch should be unreachable.
            raise ValueError

        self._power = power

    def unit_variance(self, y_pred):
        """Compute the unit variance of a Tweedie distribution
        v(y_\textrm{pred})=y_\textrm{pred}**power.

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Predicted mean.
        """
        return np.power(y_pred, self.power)

    def unit_deviance(self, y, y_pred, check_input=False):
        r"""Compute the unit deviance.

        The unit_deviance :math:`d(y,y_\textrm{pred})` can be defined by the
        log-likelihood as
        :math:`d(y,y_\textrm{pred}) = -2\phi\cdot
        \left(loglike(y,y_\textrm{pred},\phi) - loglike(y,y,\phi)\right).`

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.

        check_input : bool, default=False
            If True raise an exception on invalid y or y_pred values, otherwise
            they will be propagated as NaN.
        Returns
        -------
        deviance: array of shape (n_samples,)
            Computed deviance
        """
        p = self.power

        if check_input:
            message = ("Mean Tweedie deviance error with power={} can only be "
                       "used on ".format(p))
            if p < 0:
                # 'Extreme stable', y any realy number, y_pred > 0
                if (y_pred <= 0).any():
                    raise ValueError(message + "strictly positive y_pred.")
            elif p == 0:
                # Normal, y and y_pred can be any real number
                pass
            elif 0 < p < 1:
                raise ValueError("Tweedie deviance is only defined for "
                                 "power<=0 and power>=1.")
            elif 1 <= p < 2:
                # Poisson and Compount poisson distribution, y >= 0, y_pred > 0
                if (y < 0).any() or (y_pred <= 0).any():
                    raise ValueError(message + "non-negative y and strictly "
                                     "positive y_pred.")
            elif p >= 2:
                # Gamma and Extreme stable distribution, y and y_pred > 0
                if (y <= 0).any() or (y_pred <= 0).any():
                    raise ValueError(message
                                     + "strictly positive y and y_pred.")
            else:  # pragma: nocover
                # Unreachable statement
                raise ValueError

        if p < 0:
            # 'Extreme stable', y any realy number, y_pred > 0
            dev = 2 * (np.power(np.maximum(y, 0), 2-p) / ((1-p) * (2-p))
                       - y * np.power(y_pred, 1-p) / (1-p)
                       + np.power(y_pred, 2-p) / (2-p))

        elif p == 0:
            # Normal distribution, y and y_pred any real number
            dev = (y - y_pred)**2
        elif p < 1:
            raise ValueError("Tweedie deviance is only defined for power<=0 "
                             "and power>=1.")
        elif p == 1:
            # Poisson distribution
            dev = 2 * (xlogy(y, y/y_pred) - y + y_pred)
        elif p == 2:
            # Gamma distribution
            dev = 2 * (np.log(y_pred/y) + y/y_pred - 1)
        else:
            dev = 2 * (np.power(y, 2-p) / ((1-p) * (2-p))
                       - y * np.power(y_pred, 1-p) / (1-p)
                       + np.power(y_pred, 2-p) / (2-p))
        return dev


class NormalDistribution(TweedieDistribution):
    """Class for the Normal (aka Gaussian) distribution"""
    def __init__(self):
        super().__init__(power=0)


class PoissonDistribution(TweedieDistribution):
    """Class for the scaled Poisson distribution"""
    def __init__(self):
        super().__init__(power=1)


class GammaDistribution(TweedieDistribution):
    """Class for the Gamma distribution"""
    def __init__(self):
        super().__init__(power=2)


class InverseGaussianDistribution(TweedieDistribution):
    """Class for the scaled InverseGaussianDistribution distribution"""
    def __init__(self):
        super().__init__(power=3)


EDM_DISTRIBUTIONS = {
    'normal': NormalDistribution,
    'poisson': PoissonDistribution,
    'gamma': GammaDistribution,
    'inverse-gaussian': InverseGaussianDistribution,
}
"""
Link functions used in GLM
"""

# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# License: BSD 3 clause




class BaseLink(metaclass=ABCMeta):
    """Abstract base class for Link functions."""

    @abstractmethod
    def __call__(self, y_pred):
        """Compute the link function g(y_pred).

        The link function links the mean y_pred=E[Y] to the so called linear
        predictor (X*w), i.e. g(y_pred) = linear predictor.

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Usually the (predicted) mean.
        """

    @abstractmethod
    def derivative(self, y_pred):
        """Compute the derivative of the link g'(y_pred).

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Usually the (predicted) mean.
        """

    @abstractmethod
    def inverse(self, lin_pred):
        """Compute the inverse link function h(lin_pred).

        Gives the inverse relationship between linear predictor and the mean
        y_pred=E[Y], i.e. h(linear predictor) = y_pred.

        Parameters
        ----------
        lin_pred : array of shape (n_samples,)
            Usually the (fitted) linear predictor.
        """

    @abstractmethod
    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function h'(lin_pred).

        Parameters
        ----------
        lin_pred : array of shape (n_samples,)
            Usually the (fitted) linear predictor.
        """


class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""

    def __call__(self, y_pred):
        return y_pred

    def derivative(self, y_pred):
        return np.ones_like(y_pred)

    def inverse(self, lin_pred):
        return lin_pred

    def inverse_derivative(self, lin_pred):
        return np.ones_like(lin_pred)


class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""

    def __call__(self, y_pred):
        return np.log(y_pred)

    def derivative(self, y_pred):
        return 1 / y_pred

    def inverse(self, lin_pred):
        return np.exp(lin_pred)

    def inverse_derivative(self, lin_pred):
        return np.exp(lin_pred)


class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x)."""

    def __call__(self, y_pred):
        return logit(y_pred)

    def derivative(self, y_pred):
        return 1 / (y_pred * (1 - y_pred))

    def inverse(self, lin_pred):
        return expit(lin_pred)

    def inverse_derivative(self, lin_pred):
        ep = expit(lin_pred)
        return ep * (1 - ep)
"""
Generalized Linear Models with Exponential Dispersion Family
"""

# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause






def _safe_lin_pred(X, coef):
    """Compute the linear predictor taking care if intercept is present."""
    if coef.size == X.shape[1] + 1:
        return X @ coef[1:] + coef[0]
    else:
        return X @ coef


def _y_pred_deviance_derivative(coef, X, y, weights, family, link):
    """Compute y_pred and the derivative of the deviance w.r.t coef."""
    lin_pred = _safe_lin_pred(X, coef)
    y_pred = link.inverse(lin_pred)
    d1 = link.inverse_derivative(lin_pred)
    temp = d1 * family.deviance_derivative(y, y_pred, weights)
    if coef.size == X.shape[1] + 1:
        devp = np.concatenate(([temp.sum()], temp @ X))
    else:
        devp = temp @ X  # same as X.T @ temp
    return y_pred, devp


class GeneralizedLinearRegressor(BaseEstimator, RegressorMixin):
    """Regression via a penalized Generalized Linear Model (GLM).

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at
    fitting and predicting the mean of the target y as y_pred=h(X*w).
    Therefore, the fit minimizes the following objective function with L2
    priors as regularizer::

            1/(2*sum(s)) * deviance(y, h(X*w); s)
            + 1/2 * alpha * |w|_2

    with inverse link function h and s=sample_weight.
    The parameter ``alpha`` corresponds to the lambda parameter in glmnet.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    family : {'normal', 'poisson', 'gamma', 'inverse-gaussian'} \
            or an ExponentialDispersionModel instance, default='normal'
        The distributional assumption of the GLM, i.e. which distribution from
        the EDM, specifies the loss function to be minimized.

    link : {'auto', 'identity', 'log'} or an instance of class BaseLink, \
            default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:

        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions

    solver : 'lbfgs', default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_``.

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.
    """
    def __init__(self, *, alpha=1.0,
                 fit_intercept=True, family='normal', link='auto',
                 solver='lbfgs', max_iter=100, tol=1e-4, warm_start=False,
                 verbose=0):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a Generalized Linear Model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : returns an instance of self.
        """
        if isinstance(self.family, ExponentialDispersionModel):
            self._family_instance = self.family
        elif self.family in EDM_DISTRIBUTIONS:
            self._family_instance = EDM_DISTRIBUTIONS[self.family]()
        else:
            raise ValueError(
                "The family must be an instance of class"
                " ExponentialDispersionModel or an element of"
                " ['normal', 'poisson', 'gamma', 'inverse-gaussian']"
                "; got (family={0})".format(self.family))

        # Guarantee that self._link_instance is set to an instance of
        # class BaseLink
        if isinstance(self.link, BaseLink):
            self._link_instance = self.link
        else:
            if self.link == 'auto':
                if isinstance(self._family_instance, TweedieDistribution):
                    if self._family_instance.power <= 0:
                        self._link_instance = IdentityLink()
                    if self._family_instance.power >= 1:
                        self._link_instance = LogLink()
                else:
                    raise ValueError("No default link known for the "
                                     "specified distribution family. Please "
                                     "set link manually, i.e. not to 'auto'; "
                                     "got (link='auto', family={})"
                                     .format(self.family))
            elif self.link == 'identity':
                self._link_instance = IdentityLink()
            elif self.link == 'log':
                self._link_instance = LogLink()
            else:
                raise ValueError(
                    "The link must be an instance of class Link or "
                    "an element of ['auto', 'identity', 'log']; "
                    "got (link={0})".format(self.link))

        if not isinstance(self.alpha, numbers.Number) or self.alpha < 0:
            raise ValueError("Penalty term must be a non-negative number;"
                             " got (alpha={0})".format(self.alpha))
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("The argument fit_intercept must be bool;"
                             " got {0}".format(self.fit_intercept))
        if self.solver not in ['lbfgs']:
            raise ValueError("GeneralizedLinearRegressor supports only solvers"
                             "'lbfgs'; got {0}".format(self.solver))
        solver = self.solver
        if (not isinstance(self.max_iter, numbers.Integral)
                or self.max_iter <= 0):
            raise ValueError("Maximum number of iteration must be a positive "
                             "integer;"
                             " got (max_iter={0!r})".format(self.max_iter))
        if not isinstance(self.tol, numbers.Number) or self.tol <= 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol={0!r})".format(self.tol))
        if not isinstance(self.warm_start, bool):
            raise ValueError("The argument warm_start must be bool;"
                             " got {0}".format(self.warm_start))

        family = self._family_instance
        link = self._link_instance

        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr'],
                         dtype=[np.float64, np.float32],
                         y_numeric=True, multi_output=False)

        weights = _check_sample_weight(sample_weight, X)

        _, n_features = X.shape

        if not np.all(family.in_y_range(y)):
            raise ValueError("Some value(s) of y are out of the valid "
                             "range for family {0}"
                             .format(family.__class__.__name__))
        # TODO: if alpha=0 check that X is not rank deficient

        # rescaling of sample_weight
        #
        # IMPORTANT NOTE: Since we want to minimize
        # 1/(2*sum(sample_weight)) * deviance + L2,
        # deviance = sum(sample_weight * unit_deviance),
        # we rescale weights such that sum(weights) = 1 and this becomes
        # 1/2*deviance + L2 with deviance=sum(weights * unit_deviance)
        weights = weights / weights.sum()

        if self.warm_start and hasattr(self, 'coef_'):
            if self.fit_intercept:
                coef = np.concatenate((np.array([self.intercept_]),
                                       self.coef_))
            else:
                coef = self.coef_
        else:
            if self.fit_intercept:
                coef = np.zeros(n_features+1)
                coef[0] = link(np.average(y, weights=weights))
            else:
                coef = np.zeros(n_features)

        # algorithms for optimization

        if solver == 'lbfgs':
            def func(coef, X, y, weights, alpha, family, link):
                y_pred, devp = _y_pred_deviance_derivative(
                    coef, X, y, weights, family, link
                )
                dev = family.deviance(y, y_pred, weights)
                # offset if coef[0] is intercept
                offset = 1 if self.fit_intercept else 0
                coef_scaled = alpha * coef[offset:]
                obj = 0.5 * dev + 0.5 * (coef[offset:] @ coef_scaled)
                objp = 0.5 * devp
                objp[offset:] += coef_scaled
                return obj, objp

            args = (X, y, weights, self.alpha, family, link)

            opt_res = scipy.optimize.minimize(
                func, coef, method="L-BFGS-B", jac=True,
                options={
                    "maxiter": self.max_iter,
                    "iprint": (self.verbose > 0) - 1,
                    "gtol": self.tol,
                    "ftol": 1e3*np.finfo(float).eps,
                },
                args=args)
            self.n_iter_ = _check_optimize_result("lbfgs", opt_res)
            coef = opt_res.x

        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.
            self.coef_ = coef

        return self

    def _linear_predictor(self, X):
        """Compute the linear_predictor = `X @ coef_ + intercept_`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values of linear predictor.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=[np.float64, np.float32], ensure_2d=True,
                        allow_nd=False)
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predict using GLM with feature matrix X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values.
        """
        # check_array is done in _linear_predictor
        eta = self._linear_predictor(X)
        y_pred = self._link_instance.inverse(eta)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Compute D^2, the percentage of deviance explained.

        D^2 is a generalization of the coefficient of determination R^2.
        R^2 uses squared error and D^2 deviance. Note that those two are equal
        for ``family='normal'``.

        D^2 is defined as
        :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
        :math:`D_{null}` is the null deviance, i.e. the deviance of a model
        with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
        The mean :math:`\\bar{y}` is averaged by sample_weight.
        Best possible score is 1.0 and it can be negative (because the model
        can be arbitrarily worse).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values of target.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            D^2 of self.predict(X) w.r.t. y.
        """
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        weights = _check_sample_weight(sample_weight, X)
        y_pred = self.predict(X)
        dev = self._family_instance.deviance(y, y_pred, weights=weights)
        y_mean = np.average(y, weights=weights)
        dev_null = self._family_instance.deviance(y, y_mean, weights=weights)
        return 1 - dev / dev_null

    def _more_tags(self):
        # create the _family_instance if fit wasn't called yet.
        if hasattr(self, '_family_instance'):
            _family_instance = self._family_instance
        elif isinstance(self.family, ExponentialDispersionModel):
            _family_instance = self.family
        elif self.family in EDM_DISTRIBUTIONS:
            _family_instance = EDM_DISTRIBUTIONS[self.family]()
        else:
            raise ValueError
        return {"requires_positive_y": not _family_instance.in_y_range(-1.0)}


class PoissonRegressor(GeneralizedLinearRegressor):
    """Generalized Linear Model with a Poisson distribution.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.
    """
    def __init__(self, *, alpha=1.0, fit_intercept=True, max_iter=100,
                 tol=1e-4, warm_start=False, verbose=0):

        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         family="poisson", link='log', max_iter=max_iter,
                         tol=tol, warm_start=warm_start, verbose=verbose)

    @property
    def family(self):
        # Make this attribute read-only to avoid mis-uses e.g. in GridSearch.
        return "poisson"

    @family.setter
    def family(self, value):
        if value != "poisson":
            raise ValueError("PoissonRegressor.family must be 'poisson'!")


class GammaRegressor(GeneralizedLinearRegressor):
    """Generalized Linear Model with a Gamma distribution.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X * coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.
    """
    def __init__(self, *, alpha=1.0, fit_intercept=True, max_iter=100,
                 tol=1e-4, warm_start=False, verbose=0):

        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         family="gamma", link='log', max_iter=max_iter,
                         tol=tol, warm_start=warm_start, verbose=verbose)

    @property
    def family(self):
        # Make this attribute read-only to avoid mis-uses e.g. in GridSearch.
        return "gamma"

    @family.setter
    def family(self, value):
        if value != "gamma":
            raise ValueError("GammaRegressor.family must be 'gamma'!")


class TweedieRegressor(GeneralizedLinearRegressor):
    """Generalized Linear Model with a Tweedie distribution.

    This estimator can be used to model different GLMs depending on the
    ``power`` parameter, which determines the underlying distribution.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    power : float, default=0
            The power determines the underlying target distribution according
            to the following table:

            +-------+------------------------+
            | Power | Distribution           |
            +=======+========================+
            | 0     | Normal                 |
            +-------+------------------------+
            | 1     | Poisson                |
            +-------+------------------------+
            | (1,2) | Compound Poisson Gamma |
            +-------+------------------------+
            | 2     | Gamma                  |
            +-------+------------------------+
            | 3     | Inverse Gaussian       |
            +-------+------------------------+

            For ``0 < power < 1``, no distribution exists.

    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:

        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.
    """
    def __init__(self, *, power=0.0, alpha=1.0, fit_intercept=True,
                 link='auto', max_iter=100, tol=1e-4,
                 warm_start=False, verbose=0):

        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         family=TweedieDistribution(power=power), link=link,
                         max_iter=max_iter, tol=tol,
                         warm_start=warm_start, verbose=verbose)

    @property
    def family(self):
        # We use a property with a setter to make sure that the family is
        # always a Tweedie distribution, and that self.power and
        # self.family.power are identical by construction.
        dist = TweedieDistribution(power=self.power)
        # TODO: make the returned object immutable
        return dist

    @family.setter
    def family(self, value):
        if isinstance(value, TweedieDistribution):
            self.power = value.power
        else:
            raise TypeError("TweedieRegressor.family must be of type "
                            "TweedieDistribution!")

class TweedieGLM(TweedieRegressor):
    def summary(self, *, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
            
        y_pred = self.predict(X)
        var = self._family_instance.unit_variance(y_pred)
        self.se_ = np.sqrt(np.diag(np.linalg.pinv(X.T @ np.diag(sample_weight * var) @  X.to_numpy())))
        self.tstat_ = np.abs(self.coef_)/self.se_
        
        n, k =  X.shape
        self.df_ = n - k - self.fit_intercept
        null_dist = t(df = self.df_)
        
        self.pvalue_ = 2*null_dist.cdf(-self.tstat_)
        
        return pd.DataFrame({'ceof': self.coef_, 'se': self.se_, 'tstat': self.tstat_, 'pvalue': self.pvalue_}, index=X.columns)
