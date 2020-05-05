function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); 

J = 0;
grad = zeros(size(theta));

        pred = X*theta;
        h = sigmoid(pred);
        cst = sum(-y.*log(h) - ((1-y).*log(1-h)));
        J = (1/m) * cst + ((lambda/(2*m)) * sum( theta((2:end),1).^ 2 ));

        xa = X';
        grad(1) = (1/m) *( xa(1,:) * (h-y));
        grad(2:end) = (1/m) *( xa(2:end,:) * (h-y)) + ((lambda/m) .* theta(2:end));

end
