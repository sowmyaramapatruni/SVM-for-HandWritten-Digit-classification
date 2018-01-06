
    %% one vs all
    clearvars
    load MNIST_digit_data.mat
    
    

[W,B] = one_vs_all(images_train,labels_train,images_test,labels_test,1,0.001);   
[accuracy,y_hat,con_matrix,activation_value] = predict_all(images_test,labels_test,W,B);

[sort,IX] = sort(activation_value);
sorted_y_hat = y_hat(IX); 
sorted_y = labels_test(IX);
sorted_images = images_test(IX,:);
 
im1 = reshape(sorted_images(1, :), [28 28]);
subplot(2,5,1);
imshow(im1);
title(strcat("Predicted=",int2str(sorted_y_hat(1))," Truth=",int2str(sorted_y(1))));

im2 = reshape(sorted_images(2, :), [28 28]);
subplot(2,5,2);
imshow(im2);
title(strcat("Predicted=",int2str(sorted_y_hat(2))," Truth=",int2str(sorted_y(2))));

im3 = reshape(sorted_images(3, :), [28 28]);
subplot(2,5,3);
imshow(im3);
title(strcat("Predicted=",int2str(sorted_y_hat(3))," Truth=",int2str(sorted_y(3))));

im4 = reshape(sorted_images(4, :), [28 28]);
subplot(2,5,4);
imshow(im4);
title(strcat("Predicted=",int2str(sorted_y_hat(4))," Truth=",int2str(sorted_y(4))));

im5 = reshape(sorted_images(5, :), [28 28]);
subplot(2,5,5);
imshow(im5);
title(strcat("Predicted=",int2str(sorted_y_hat(5))," Truth=",int2str(sorted_y(5))));

im6 = reshape(sorted_images(6, :), [28 28]);
subplot(2,5,6);
imshow(im6);
title(strcat("Predicted=",int2str(sorted_y_hat(6))," Truth=",int2str(sorted_y(6))));

im7 = reshape(sorted_images(7, :), [28 28]);
subplot(2,5,7);
imshow(im7);
title(strcat("Predicted=",int2str(sorted_y_hat(7))," Truth=",int2str(sorted_y(7))));

im8 = reshape(sorted_images(8, :), [28 28]);
subplot(2,5,8);
imshow(im8);
title(strcat("Predicted =",int2str(sorted_y_hat(8))," Truth=",int2str(sorted_y(8))));

im9 = reshape(sorted_images(9, :), [28 28]);
subplot(2,5,9);
imshow(im9);
title(strcat("Predicted=",int2str(sorted_y_hat(9))," Truth=",int2str(sorted_y(9))));

im10 = reshape(sorted_images(10, :), [28 28]);
subplot(2,5,10);
imshow(im10);
title(strcat("Predicted=",int2str(sorted_y_hat(10))," Truth=",int2str(sorted_y(10))));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ W,B ] = one_vs_all( images_train,labels_train,images_test,labels_test,epoch, C )
    for i = 0 : 9
       classifier = i
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


 function [ accuracy,y_hat ,con_matrix,activation_value] = predict_all( images_test,labels_test,W,b )

 error = 0;
 con_matrix = zeros(10,10);
 activation_value = zeros(1,10000);
 sizeTest = size(labels_test);
 for i = 1:sizeTest
     value = images_test(i,:)*transpose(W) + b;
     [M,I] = max(value);
     activation_value(i) = M;
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


 
    