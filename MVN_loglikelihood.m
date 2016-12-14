function [ll] = MVN_loglikelihood(X, model)
% Calculate the log likelihood of X, given an ICA model.
% 
% The model assumes each column of x is independently generated by single
% 0-mean gaussian.
%
% Arguments
%  X - A DxM matrix, whose every column corresponds to a patch in D
%      dimensions (typically D=64).
%  model - A struct with field:
%           cov - A DxD covariane matrix.
%

D = size(X, 1);
model.means = zeros(1, D);
model.covs = model.cov;
model.mix = 1;

ll = GMM_loglikelihood(X, model);