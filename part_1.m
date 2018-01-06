function [ W,b ] = grad_svm_binary( images_train,labels_train,images_test,labels_test,epoch, C )

X = images_train;
Y = labels_train;
W = zeros(1,784);
sizeDataTrain=size(labels_train,1);
b = 0;
   while epoch > 0
      g = zeros(1,784);
      g_b = 0;
      accuracy_iter = zeros(sizeDataTrain,1);
      for i = 1:sizeDataTrain
          y_hat = dot(X(i,:),W) + b;
          if y_hat*Y(i) <= 1
             g = g + Y(i)*X(i,:);
             g_b = g_b + Y(i); 
             g = g-C*W;
             W = W + (1/i)*g;
             b = b + (1/i)*g_b;            
          end        
      end
      epoch = epoch - 1;
   end
 
end

