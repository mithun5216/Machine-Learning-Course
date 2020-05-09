function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);
       
            a1 = [ones(m,1) X];
            pred = a1*Theta1';
            h = sigmoid(pred);
            a2 = [ones(m,1) h];
            a3 = a2*Theta2';
            
            [val,ind] = max(a3,[],2);
            p = ind;


end
