function x = emailFeatures(word_indices)

n = 1899;

x = zeros(n, 1);


        for i=1:n
            for j=1:length(word_indices)
                
                if i == word_indices(j) 
                     x(i) = 1;       
                end
                                   
            end
        end
 

end
