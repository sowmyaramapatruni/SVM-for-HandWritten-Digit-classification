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