function [model] = learn_MVN(X, options)
% Learn parameters for a 0-mean multivariate normal model for X.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in mixture.
%   options - options for learn_GMM (optional).
% Returns:
%   model - a struct with 3 fields:
%            cov - DxD covariance matrix.
%

model = struct('cov', cov(X', 1));
