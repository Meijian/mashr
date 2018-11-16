.fixfix <- function(fix, ngauss) {
    if (is.null(fix)) {
        fix <- rep(0, ngauss)
    } else if (fix == FALSE | fix == TRUE) {
        fix <- rep(as.integer(fix), ngauss)
    } else if (length(fix) != ngauss) {
        warning("Dimension of fix* input does not match data (set all to the first entry)!")
        fix <- rep(as.integer(fix[0]), ngauss)
    } else {
        fix <- as.integer(fix)
    }
    return(fix)
}

#' @title Density estimation using Gaussian mixtures in the presence of
#'   noisy, heterogeneous and incomplete data
#'
#' @description We present a general algorithm to infer a
#'   d-dimensional distribution function given a set of heterogeneous,
#'   noisy observations or samples. This algorithm reconstructs the
#'   error-deconvolved or "underlying" distribution function common to
#'   all samples, even when the individual samples have unique error and
#'   missing-data properties. The underlying distribution is modeled as
#'   a mixture of Gaussians, which is completely general. Model
#'   parameters are chosen to optimize a justified, scalar objective
#'   function: the logarithm of the probability of the data under the
#'   error-convolved model, where the error convolution is different for
#'   each data point. Optimization is performed by an Expectation
#'   Maximization (EM) algorithm, extended by a regularization technique
#'   and "split-and-merge" procedure. These extensions mitigate problems
#'   with singularities and local maxima, which are often encountered
#'   when using the EM algorithm to estimate Gaussian density mixtures.
#'
#' @param ydata [ndata,dy] matrix of observed quantities.
#' 
#' @param ycovar [ndata,dy] / [ndata,dy,dy] / [dy,dy,ndata] matrix,
#'   list or 3D array of observational error covariances (if [ndata,dy]
#'   then the error correlations are assumed to vanish).
#'
#' @param xamp [ngauss] array of initial amplitudes (*not* [1,ngauss]).
#' 
#' @param xmean [ngauss,dx] matrix of initial means.
#' 
#' @param xcovar [ngauss,dx,dx] list of matrices of initial covariances.
#' 
#' @param projection [ndata,dy,dx] list of projection matrices.
#' 
#' @param weight [ndata] array of weights to be applied to the data points.
#' 
#' @param logweight (bool) if True, weight is actually
#'   log(weight).
#'
#' @param fixamp None, True/False, or list of bools.
#' 
#' @param fixmean None, True/False, or list of bools.
#' 
#' @param fixcovar None, True/False, or list of bools.
#' 
#' @param tol (double) Tolerance for convergence.
#' 
#' @param maxiter Maximum number of iterations to perform.
#' 
#' @param w (double) covariance regularization parameter (of the
#'   conjugate prior).
#' 
#' @param logfile Basename for several logfiles (_c.log has output
#'   from the c-routine; _loglike.log has the log likelihood path of all
#'   the accepted routes, i.e. only parts which increase the likelihood
#'   are included, during splitnmerge).
#'
#' @param splitnmerge (int, default=0) depth to go down the
#'   splitnmerge path.
#' 
#' @param (Bool) Use the maximum number of split 'n' merge steps,
#'   K*(K-1)*(K-2)/2.
#' 
#' @param likeonly (Bool) Only compute the total log likelihood of the
#'   data.
#'
#' @return A list object with the following elements:
#' \item{avgloglikedata}{avgloglikedata after convergence}
#' \item{xamp}{updated xamp}
#' \item{xmean}{updated xmean}
#' \item{xcovar}{updated xcovar}
#' 
#' @author Jo Bovy, David W. Hogg & Sam T. Roweis.
#'
#' @references
#' Inferring complete distribution functions from noisy, heterogeneous
#' and incomplete observations Jo Bovy, David W. Hogg, & Sam
#' T. Roweis, Submitted to AOAS (2009) [arXiv/0905.2979]
#'
#' @export
#' 
extreme_deconvolution <- function(ydata, ycovar, xamp, xmean, xcovar, projection = NULL, 
    weight = NULL, fixamp = NULL, fixmean = NULL, fixcovar = NULL, tol = 1e-06, maxiter = 1e+09, 
    w = 0, logfile = NULL, splitnmerge = 0, maxsnm = FALSE, likeonly = FALSE, logweight = FALSE) {
    ndata <- dim(ydata)[1]
    dataDim <- dim(ydata)[2]
    ngauss <- length(xamp)
    gaussDim <- dim(xmean)[2]
    if (typeof(ycovar) == "list") {
        tycovar <- unlist(lapply(ycovar, t))
        diagerrors <- FALSE
    } else if (length(dim(ycovar)) == 3) {
        tycovar <- apply(ycovar, 3, t)
        diagerrors <- FALSE
    } else {
        # a matrix
        tycovar <- t(ycovar)
        diagerrors <- TRUE
    }
    fixamp <- .fixfix(fixamp, ngauss)
    fixmean <- .fixfix(fixmean, ngauss)
    fixcovar <- .fixfix(fixcovar, ngauss)
    avgloglikedata <- 0
    # 
    if (is.null(logfile)) {
        clog <- 0
        clog2 <- 0
        n_clog <- 0
        n_clog2 <- 0
    } else {
        clog <- charToRaw(paste(logfile, "c.log", sep = "_"))
        n_clog <- length(clog)
        clog2 <- charToRaw(paste(logfile, "loglike.log", sep = "_"))
        n_clog2 <- length(clog2)
    }
    # 
    if (maxsnm) 
        splitnmerge <- ngauss * (ngauss - 1) * (ngauss - 2)/2
    if (is.null(projection)) {
        noprojection <- TRUE
        projection <- list()
    } else {
        noprojection <- FALSE
    }
    if (is.null(weight)) {
        noweight <- TRUE
        logweights <- array(0)
    } else if (!logweight) {
        noweight <- FALSE
        logweights <- log(weight)
    } else {
        noweight <- FALSE
        logweights <- weight
    }
    # 
    res <- .C("proj_gauss_mixtures_IDL", as.double(as.vector(t(ydata))), as.double(as.vector(tycovar)),
        as.double(as.vector(unlist(lapply(projection, t)))), as.double(as.vector(logweights)), 
        as.integer(ndata), as.integer(dataDim), xamp = as.double(as.vector(t(xamp))), 
        xmean = as.double(as.vector(t(xmean))), xcovar = as.double(as.vector(unlist(lapply(xcovar, 
            t)))), as.integer(gaussDim), as.integer(ngauss), as.integer(as.vector(fixamp)), 
        as.integer(as.vector(fixmean)), as.integer(as.vector(fixcovar)), avgloglikedata = as.double(as.vector(avgloglikedata)), 
        as.double(tol), as.integer(maxiter), as.integer(likeonly), as.double(w), 
        as.integer(as.vector(clog)), as.integer(n_clog), as.integer(splitnmerge), 
        as.integer(as.vector(clog2)), as.integer(n_clog2), as.integer(noprojection), 
        as.integer(diagerrors), as.integer(noweight), PACKAGE = "ExtremeDeconvolution")
    # 
    xmean <- matrix(res$xmean, dim(xmean), byrow = TRUE)
    start <- 1
    end <- 0
    for (i in 1:length(xcovar)) {
        end <- end + prod(dim(xcovar[[i]]))
        xcovar[[i]] <- matrix(res$xcovar[start:end], dim(xcovar[[i]]), byrow = TRUE)
        start <- end + 1
    }
    return(list(xmean = xmean, xamp = res$xamp, xcovar = xcovar, avgloglikedata = res$avgloglikedata))
} 
