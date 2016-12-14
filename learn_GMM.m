function [theta, LL] = learn_GMM(X, K, params0, options)
% Learn parameters for a gaussian mixture model via EM.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in mixture.
%   params0 - An optional struct with intialization parameters. Has 3
%             optional fields:
%               means - a KxD matrix whose every row corresponds to a
%                       component mean.
%               covs - A DxDxK array, whose every page is a component
%                      covariance matrix.
%               mix - A Kx1 mixture vector (should sum to 1).
%             If not given, a random starting point is generated.
%   options - Algorithm options struct, with fields:
%              learn - A struct of booleans, denoting which parameters
%                      should be learned: learn.means, learn.covs and
%                      learn.mix. The default is that given parameters
%                      (in params0) are not learned.
%              max_iter - maximum #iterations. Default = 100.
%              thresh - if previous_LL * thresh > current_LL,
%                       algorithm halts. default = 1.01.
%              verbosity - either 'none', 'iter' or 'plot'. default 'none'.
% Returns:
%   params - A struct with learned parameters (fields):
%               means - a KxD matrix whose every row corresponds to a
%                       component mean.
%               covs - A DxDxK arramodel = struct('cov', cov(X', 1));y, whose every page is a component
%                      covariance matrix.
%               mix - A Kx1 mixture vector.
%   LL - log likelihood history
%
% =========================================================================
% This is an optional file - use it if you want to implement a single EM
% algorithm
% =========================================================================
%
EPS = 1e-10;
if ~exist('params0', 'var')
    params0 = struct(); 
end
[theta, default_learn] = get_params0(X, K, params0);

if ~exist('options', 'var') 
    options = struct(); 
end
options = organize_options(options, default_learn);

means = theta.means;
covs = theta.covs;
mix = theta.mix;

LL = [];
curLL = -inf; % todo maybe 0

D = size(X, 1);
M = size(X,2);

logDensXgivenH = zeros(M,K);

scale = 0;
if isfield(options.learn, 'scale')
   scale = 1; 
   covX = cov(X');
   invCovX = inv(covX);
   scale_factors = zeros(K, 1);
   for k = 1:K
       scale_factors(k) = rand() + EPS;
       covs(:,:,k) = scale_factors(k) * covX;
   end
end

for n_iter = 1:options.max_iter
    % E step
    for k = 1:K
        logDensXgivenH(:,k) = log_mvnpdf(X', means(k,:), covs(:,:,k));    
    end
    
    %logDensXgivenH = logDensXgivenH + EPS;
    
    logmatProbH = repmat(log(mix), 1, M)';
    logProbX = logsum(logDensXgivenH + logmatProbH, 2);
    logmatInvProbX = repmat(-logProbX, 1, K);
    logProbXgivenHdivProbX = logDensXgivenH + logmatInvProbX;
    logProbHgivenX = logProbXgivenHdivProbX + logmatProbH;  
    
    probHgivenX = exp(logProbHgivenX);
    
    % M step
    sumColsHgivenX = sum(probHgivenX, 1);
    for k = 1:K
        % updating scale factors
        if scale
            tmpScale = 0;
            for i = 1:M
                tmpScale = tmpScale + (probHgivenX(i, k) * X(:,i)' * invCovX * X(:,i) / (D * sumColsHgivenX(k)));
            end
            scale_factors(k) = tmpScale;
            covs(:,:,k) = scale_factors(k) * covX;
        end
        
        % updating means
        if options.learn.means
            means(k,:) = (sum(repmat(probHgivenX(:,k), 1, D) .* (X'), 1)) / sumColsHgivenX(k);
        end
        
        % updating covs
        if options.learn.covs
            tmpCov = zeros(D, D);
            for i = 1:M
                tmpCov = tmpCov + (probHgivenX(i,k) * (X(:,i) - means(k,:)') * (X(:,i) - means(k,:)')');
            end
            covs(:,:,k) = tmpCov / sumColsHgivenX(k);
        end
    
        % updaing mix
        if options.learn.mix
            mix(k) = sumColsHgivenX(k) / sum(probHgivenX(:));
            %TODO maybe change to 1/M where M is the number of examples
        end
    end
    
    % udpating theta
    theta.means = means;
    theta.covs = covs;
    theta.mix = mix;
    
    % calc logliklihood
    prevLL = curLL;
    curLL = GMM_loglikelihood(X, theta);
    LL = [LL curLL];
    
    % check convergence
    if curLL < prevLL * options.threshold
        break;
    end
end
   
end


function [params0, default_learn] = get_params0(X, K, params0)
% organizes the params0 struct and output the starting point of the
% algorithm - "params0".
default_learn.mix = false;
default_learn.means = false;
default_learn.covs = false;

[D,M] = size(X);

if ~isfield(params0, 'means')
    default_learn.means = true;
    params0.means = X(:,randi(M, [1,K]))';
    params0.means = params0.means + nanstd(X(:))*randn(size(params0.means));
end

if ~isfield(params0, 'covs')
    default_learn.covs = true;
    params0.covs = nan(D,D,K);
    for k = 1:K
        params0.covs(:,:,k) = nancov(X(:,randi(M, [1,10]))');
    end
end

if ~isfield(params0, 'mix')
    default_learn.mix = true;
    params0.mix = rand(K,1);
    params0.mix = params0.mix / sum(params0.mix);
end

end



    



    

function [options] = organize_options(options, default_learn)
%organize the options.
if ~isfield(options, 'threshold') options.threshold = 1.01; end
if ~isfield(options, 'max_iter') options.max_iter = 100; end
if ~isfield(options, 'verbosity') options.verbosity = 'none'; end
if ~isfield(options, 'learn') options.learn = default_learn;
else
    if ~isfield(options.learn, 'means') options.learn.means = default_learn.means; end;
    if ~isfield(options.learn, 'covs') options.learn.covs = default_learn.covs; end;
    if ~isfield(options.learn, 'mix') options.learn.mix = default_learn.mix; end;
end
end