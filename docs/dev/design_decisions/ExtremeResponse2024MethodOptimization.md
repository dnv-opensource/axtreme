# Extreme Response 2024 method

## Problem
In the method it requires finding a percentile of a (not straight forward) probability distribution.
In particular this cdf is a linear mixture of cdfs of the same type, but with different parameters.
In addition there are multiple decoupled versions of this problems that should be solved (in parallel).
Finding this percentile could be done through the inverse cdf of the mixture distribution. However, this is very difficult in practice. Therefore we instead use root finding / optimization methods to find the value of the inverse cdf on for the specific percentile we are interested in.


## Choice of method
### scipy
One option would be to use scipy.optimize. However, this will make gradient tracking and use of GPU not possible. Copying data between cpu and gpu is slow and should be avoided if possible. Also scypy.optimize does not have a good option for solving several decoupled problems in parallel.

### pytorch autograd
We cannot use pytorch gradient calculation for this inner optimization loop since it will mess with the outer optimization that we care about when this is used as part in e.g. acquisition functions. So we cannot use pytorch autograd for this.

### pytorch implementation of methods

To enable use of GPU and gradient we can use pytorch built in method to implement optimization/root finding methods like newton's method and secant method.

In addition since we are implementing this ourselves for a specific problem (cdfs) we can make assumptions about the problem solved to improve the optimization. We can therefore make the implementations be able to solve multiple decoupled problems in parallel to speed up the optimization. These multiple decoupled problems are also likely to have similar solutions since these cdfs are sampled from the same GP.

The assumptions made:
 - Since we are optimizing a cdf, we know that the derivative is a pdf which is always non-negative.
 - There exists only a single root since objective is a cdf and therefore non-decreasing. So if a solution is found we know it is the solution we are looking for.
 - Newton
   - Use the pdf directly as the derivative of the objective
   - The method converges when the optimization is started "close enough" to the optimum
   - The method struggles in the regions where the cdf is flat (pdf close to zero).
 - Secant
    - Needs to have to starting points where they are on either side of the optimum
    - Since cdf in increasing function we know we can always find this
    - The method converges when the optimization is started "close enough" to the optimum
    - The method struggles in the regions where the cdf is flat (pdf close to zero).
 - Bisect
    - Needs two starting points at either side of the optimum (this can be found)
    - Converges slowly, but will always (as far as i know) converge after some number of iterations. However, the number of required iterations can eb very large if high accuracy is required


## Resulting method

The resulting method ended up with was a combination of the above.
The pytorch implementations of newton, secant and bisect are used together to find the percentile. These methods can fail or not find the optimum in the amount of iterations we allow. So therefore these methods are restarted multiple times where only the problems we have not found the solution for is optimized further. Since we assume the solutions of the equations are similar we restart the methods at points similar to the solutions that are found for the other problems.

If after a set amount of restarts if there are still some problems that have not converged we use scipy.optimize.root_scalar to solve the remaining problems that have not converged.

Until now everything has been done using torch.no_grad. So if gradient tracking is wanted we can run a single step of newtons method starting at the solution already found. Since we are using newtons method with pytorch functions this will reconnect the gradient.
