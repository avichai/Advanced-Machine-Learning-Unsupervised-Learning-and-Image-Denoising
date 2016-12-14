function [xhat] = GMM_denoise(y, gmm, noise)
% Denoises every column in y, assuming a gaussian mixture model and white
% noise.
% 
% The model assumes that y = x + noise where x is generated from a GMM.
%
% Arguments
%  y - A DxM matrix, whose every column corresponds to a patch in D
%      dimensions (typically D=64).
%  gmm - The mixture model, with 4 fields:
%          means - A KxD matrix where K is the number of components in
%                  mixture and D is the dimension of the data.
%          covs - A DxDxK array whose every page is a covariance matrix of
%                 the corresponding component.
%          mix - A Kx1 vector with mixing proportions.
%  noise - the std of the noise in y.
%

% todo add if noise = 0

means = gmm.means;
covs = gmm.covs;
mix = (gmm.mix);

K = length(mix);
[D, M] = size(y);

xhat = zeros(D, M);

invVarNoise = 1/(noise^2);

new_covs = covs + repmat(noise^2 * eye(D), [1 1 K]);
gm_dist_obj = gmdistribution(means, new_covs, mix);
pdf_y = pdf(gm_dist_obj, y');
h_given_y = zeros(M, K);
expect_x_given_y_h = zeros(D, M, K);

for k = 1:K
    % calc the expectation of x given y and h
    inv_cov = inv(covs(:,:,k)); 
    left_parentheses = inv(inv_cov + invVarNoise * eye(D));
    lhs_v = left_parentheses * (inv_cov * means(k,:)');
    lhs = repmat(lhs_v, [1 M]);
    rhs = left_parentheses * ((1/noise^2) * y);
    expect_x_given_y_h(:,:,k) = lhs + rhs;
    
    % calc the probability of h given y
    pdf_y_given_h = mvnpdf(y', means(k,:), new_covs(:,:,k));
    %pdf_y(pdf_y == 0) = pdf_y(pdf_y == 0) + 1e-10;
    h_given_y(:,k) = (mix(k) * pdf_y_given_h) ./ pdf_y;
end


for m = 1:M
    for k = 1:K
        xhat(:,m) = xhat(:,m) + expect_x_given_y_h(:,m,k) * h_given_y(m, k);
    end
end