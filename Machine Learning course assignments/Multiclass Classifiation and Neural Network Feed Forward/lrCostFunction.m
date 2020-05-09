function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

        pred = X*theta;
        h = sigmoid(pred);
        s = -sum((y.*log(h))+((1-y).*(log(1-h))));
        J = ((1/m) * s) + ((lambda/(2*m))*sum(theta(2:end,1).^2));

        x = X';
        diff = h - y;
        grad(1) = (1/m) .* (x(1,:) * diff );
        grad(2:end) = ((1/m) .* (x(2:end,:) * diff)) + ((lambda/m) .* theta(2:end));


grad = grad(:);

end
