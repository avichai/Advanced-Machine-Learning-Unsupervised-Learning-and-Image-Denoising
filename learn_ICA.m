function model = learn_ICA(X, K, options)
% Learn parameters for a complete invertible ICA model.
%
% We learn a matrix P such that X = P*S, where S are D independent sources
% And for each of the D coordinates we learn a mixture of K (univariate)
% 0-meanfunction model = learn_ICA(X, K, options)
% Learn parameters for a complete invertible ICA model.
%
% We learn a matrix P such that X = P*S, where S are D independent sources
% And for each of the D coordinates we learn a mixture of K (univariate)
% 0-mean gaussians via EM.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in a mixture.
%   options - options for learn_GMM (optional).
% Returns:
%   model - A struct with 3 fields:
%           P - mixing matrix of sources (P: D ind. sources -> D signals)
%           vars - a DxK matrix whose (d,k) element correponsds to the
%                  variance of the k'th component in dimension d.
%           mix - a DxK matrix whose (d,k) element correponsds to the
%                 mixing weight of the k'th component in dimension d.
%ad

D = size(X, 1);
covX = cov(X');
[P, ~] = eig(covX);
S = P' * X;

% set params0
params0.means = zeros(K,1);

% setting options
options.learn.means = false;
options.learn.covs = true;
options.mix = true;

vars = zeros(D, K);
mix = zeros(D, K);

% calc each component independently
for i = 1:D
	[theta, ~] = learn_GMM(S(i,:), K, params0, options);
	vars(i,:) = theta.covs;
	mix(i,:) = theta.mix';
end

% set model
model.P = P;
model.vars = vars;
model.mix = mix;






