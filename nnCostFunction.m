function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for the 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2)); 

X=[ones(m,1) X];


for i=1:m,
    
	a2=sigmoid(X(i,:)*Theta1');
    h=sigmoid([ones(1,1) a2]*Theta2')';
	Y=1:1:num_labels==y(i);
	J+=sum(Y.*log(h)'+(1-Y).*log(1-h)');
	dl=h-Y'; % delta for the output layer
    d2=Theta2'*dl.*([ones(1,1) a2].*(1-[ones(1,1) a2]))';
	
	d2=d2(2:end);
	Theta2_grad=Theta2_grad+(dl*[ones(1,1) a2]);
	Theta1_grad=Theta1_grad+(d2*X(i,:));
	
 end;

 Theta2_grad=Theta2_grad/m;
 Theta1_grad=Theta1_grad/m;
 
 %regularize them
 Theta2_grad=Theta2_grad+(lambda*Theta2/m);
 Theta2_grad(:,1)=Theta2_grad(:,1)-(lambda*Theta2(:,1)/m);
 Theta1_grad=Theta1_grad+(lambda*Theta1/m);
 Theta1_grad(:,1)=Theta1_grad(:,1)-(lambda*Theta1(:,1)/m);

J=-J/m; 
J+=(sum(sum(Theta1.^2))+sum(sum(Theta2.^2))-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2))*lambda/(m*2);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
