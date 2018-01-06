clearvars
load MNIST_digit_data.mat


 %% loading train data
 classifier1 = 6;
 classifier2 = 1;
    A=find(labels_train == classifier1); 
    B=find(labels_train == classifier2);
    C = union(A,B);
    labels_train1 = labels_train(C);
    images_train1 = images_train(C,:);
    labels_train1 = labels_train1(1:1000);
    images_train1 = images_train1(1:1000,:);
    
    %% Mapping labels into 1 and -1 values
    for k=1:1000
       if labels_train1(k)==classifier1
           labels_train1(k)= -1;
       else
           labels_train1(k)=1;
       end 
    end
    
    %% loading test data 
    A=find(labels_test==classifier1); 
    B=find(labels_test==classifier2);
    C = union(A,B);
    labels_test1 = labels_test(C);
    images_test1 = images_test(C,:);
    for k=1:length(C)
       if labels_test1(k)==classifier1
           labels_test1(k)=-1;
       else
           labels_test1(k)=1;
       end 
    end
    
        
[W,b] = grad_svm_binary_plot(images_train1,labels_train1,images_test1,labels_test1,300,0.01);

%%%%%%%%%%%%%%%%%%%%%%%%%Supportingfunctions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ accuracy ] = Predict( images_test,labels_test,W,b )

 error = 0;
 length = size(labels_test);
 for i = 1:length
     value = dot(images_test(i,:),W) + b;
     if labels_test(i)*value < 1
         error = error+1;
     end
 end
 accuracy = 100 *((length-error)/length)

end

function [ W,b ] = grad_svm_binary_plot( images_train,labels_train,images_test,labels_test,epoch, C )

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
                              
          end
          accuracy_iter(i) = Predict(images_test,labels_test,W,b);
          
      end
        g = g-C*W;
        W = W + (1/i)*g;
        b = b + (1/i)*g_b; 
        
        
      epoch = epoch - 1;
   end

     x= linspace(1,sizeDataTrain,sizeDataTrain);
     y=accuracy_iter;
     plot(x,y);
     title('graph of accuracy versus number of iteration');
     xlabel('number of iterations');
     ylabel('accuracy');
     
end

