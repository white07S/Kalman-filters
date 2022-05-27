# Kalman-filters
Kalman filters implementation in Financial models for correlation and Linear regression
# Kalman filter is a great tool and here we are going to apply into Linear Regression

# Some docs 
Linear regression : https://en.wikipedia.org/wiki/Linear_regression

# Analysis

![Alpha](https://github.com/white07S/Kalman-filters/blob/main/assets/alpha.png)

![Beta](https://github.com/white07S/Kalman-filters/blob/main/assets/beta.png)


# Comments

Looking at the two plots above, we can see that the curves are very similar. An important thing to notice is that the OLS is shiftet on the right i.e. OLS is delayed! 

The Kalman estimates can reflect much better the true (spot) value of the state.

The filter is very robust with respect to initial guesses. If, for instance, you try to set P = 100 * np.eye(2), you will see that it will converge quickly to a "reasonable" value.

In the matrix H[None,i] I had to use the keyword None in order to maintain the information of the array dimensions, H[None,0].shape = (1,2). This is a bad feature of the language. Python automatically removes dimensions of a subarray i.e. H[0].shape = (2,).

K @ H[None,i] is an outer product of dimension 

I used inv(S) to compute the inverse of S. In the notebook A1 I repeated many times that inverting matrices is bad. But for the Kalman filter, since usually S has small rank (here S is a scalar), we can make an exception :)

# PROS

The Kalman filter does not need an entire dataset (or time window) to estimate the current state. It just needs ONE measurement!
The OLS estimate, instead, depends on the length of the time window, and it never reflects the spot value of the state because it is affected by past data.

# CONS

The Kalman filter performances are strongly dependent on the process and measurement errors!!
These values are not always measurable, and are hard to calibrate.

![Beta](https://github.com/white07S/Kalman-filters/blob/main/assets/log.png)

![Beta](https://github.com/white07S/Kalman-filters/blob/main/assets/mle.png)

# Conclusion
The error comparison above is only indicative. Since the OLS error can be set arbitrarily small by simply choosing a bigger time window (the more data we have, the higher will be).
