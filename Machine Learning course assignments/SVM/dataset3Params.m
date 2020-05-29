function [C, sigma] = dataset3Params(X, y, Xval, yval)


C = 1;
sigma = 0.3;

    value = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    error_min = inf;
    
    for i = value
        for j = value
            model = svmTrain(X, y, i, @(x1, x2)gaussianKernel(x1, x2, j));
            predictions = svmPredict(model,Xval);
            error = mean(double(predictions~=yval));     
            
            if (error<= error_min)
                C_final = i;
                Sigma_final = j;
                error_min = error;
                
            end
        end
    end

    C = C_final;
    sigma = Sigma_final;

end
