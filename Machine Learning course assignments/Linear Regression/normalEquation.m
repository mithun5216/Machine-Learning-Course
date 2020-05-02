function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);
sa = 0;

sa = pinv(X'*X);
theta = sa * (X' * y);

end
