function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));


        I = eye(num_labels);
        Y = zeros(m,num_labels);
        for i= 1:m
            Y(i,:) = I(y(i),:);
        end
        
        %Feed Forward
        
        a1 =[ones(m,1) X];
        z1 = a1 * Theta1';
        a2 = sigmoid(z1);
        a2 = [ones(m,1) a2];
        z2 = a2* Theta2';
        a3 = sigmoid(z2);
        
        %Cost Function
        
        sumoferrors =sum( sum(-Y.*log(a3) - ((1-Y) .*log(1-a3) )));
        J = (1/m) .* sumoferrors ;
            
        J = J + ((lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2,2)) + sum(sum(Theta2(:,2:end).^2,2))));


        %Back Propogation
        
        for t=1:m
            
           A1 = [1 X(t,:)];
           Z2 = A1 * Theta1';
           A2 = [1 sigmoid(Z2)];           
           Z3 = A2 * Theta2';
           A3 = sigmoid(Z3);
           
           S3 = A3 - Y(t,:);
           S2 = (S3 * Theta2) .* sigmoidGradient([1 Z2]);
           S2 = S2(1,2:end);
           
           Delta1 = Delta1 + (S2'*A1);
           Delta2 = Delta2 + (S3'*A2);
          
                  
        end
        
         Theta1_grad(:,1) = Delta1(:,1)/m;
         Theta1_grad(:,2:end) = (Delta1(:,2:end)/m)+ ((lambda/m) .* Theta1(:,2:end));
          Theta2_grad(:,1) = Delta2(:,1)/m;
          Theta2_grad(:,2:end) = (Delta2(:,2:end)/m) + ((lambda/m) .* Theta2(:,2:end));
          
          
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
