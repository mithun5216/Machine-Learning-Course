function[X_norm,mu,sigma] = featureNormalize(x)

mu = zeros(1,size(x,2))
sigma = zeros(1,size(x,2)) 

mu = mean(X);
sigma = std(X);
m = size(x,1);

for 1:m:

	X_norm(1,:) = (X(1,:) - mu) ./ sigma ; 

end

end