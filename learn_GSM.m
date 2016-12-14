function [model] = learn_GSM(X, K, options)
% Learn parameters for a gaussian scaling mixture model for X via EM
%
% GSM components share the variance, up to a scaling factor, so we only
% need to learn scaling factors c_1.. c_K and mixture proportions
% alpha_1..alpha_K.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in mixture.
%   options - options for learn_GMM (optional).
% Returns:
%   model - a struct with 3 fields:
%           mix - Mixture proportions.
%           covs - A DxDxK array, whose every page is a scaled covariance
%                  matrix according to scaling parameters.
%           means - K 0-means.
%

%cov = cov(X');
%covs = repmat(sig, [1 1 K]);
%for k = 1:K
%    covs(:,:,k) = covs(:,:,k)*(10*rand());
%end

D = size(X, 1);
means = zeros(K, D);
params0.means = means;
params0.covs = zeros(D, D, K);
options.learn.means= 0;
options.learn.covs = 0;
options.learn.mix = 1;
options.learn.scale = 1;

[model, ll] = learn_GMM(X,K,params0,options);
