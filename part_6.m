
    %% one vs all
    clearvars
    load MNIST_digit_data.mat
    
    

[W,B] = one_vs_all(images_train,labels_train,images_test,labels_test,1,0.01);   
[accuracy,con_matrix] = predict_all(images_test,labels_test,W,B);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Supporting functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
function [ W,B ] = one_vs_all( images_train,labels_train,images_test,labels_test,epoch, C )
    for i = 0 : 9
       classifier = i;
       labels_train_one_vs_all = ones(size(labels_train));
       labels_test_one_vs_all = ones(size(labels_test));
       labels_train_one_vs_all(labels_train ~= classifier) = -1;
       labels_test_one_vs_all(labels_test ~= classifier) = -1;
       
       [w,b] = grad_svm_binary(images_train,labels_train_one_vs_all,images_test,labels_test_one_vs_all,epoch,C);
       W(i+1,:) = w;
       B(i+1) = b;  
    end
end

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


 function [ accuracy,con_matrix] = predict_all( images_test,labels_test,W,b )

 error = 0;
 con_matrix = zeros(10,10);
 sizeTest = size(labels_test);
 for i = 1:sizeTest
     value = images_test(i,:)*transpose(W) + b;
     [M,I] = max(value);
     y_hat(i) = I-1;
     con_matrix(y_hat(i)+1,labels_test(i)+1) = con_matrix(y_hat(i)+1,labels_test(i)+1) +1;
     if y_hat(i) ~= labels_test(i)
         error = error+1;
     end
 end
 
 for k = 1:10
     sum_value = sum(con_matrix(k,:));
     con_matrix(k,:) = con_matrix(k,:)/sum_value;
 end
 
 accuracy = 100 *((sizeTest-error)/sizeTest)

end


 
    