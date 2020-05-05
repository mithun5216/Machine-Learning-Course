function [J, grad] = costFunction(theta, X, y)

m = length(y); 

J = 0;
grad = zeros(size(theta));

            pred = X * theta;
            h = sigmoid(pred);
            cost = sum(-y.*log(h) - (1-y).*log(1-h));
            J = (1/m) .* cost;
                      
            grad = (1/m) * (X' * (h - y ));

end
