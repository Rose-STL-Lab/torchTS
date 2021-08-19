% Numerical solution to the Lorenz63 equation using the 4th order Runge-Kutta method
h = 0.01;
t_max = 2000;

t_vals = 0:h:t_max;

[~,size_t] = size(t_vals);

x = zeros(3,size_t);
x(1,1) = 1;

A = [0,0,0,0;1/2,0,0,0;0,1/2,0,0;0,0,1,0];
b = [1/6, 1/3, 1/3, 1/6];
c = [0, 1/2, 1/2, 1];

for n=1:size_t-1
    k_1 = x(:,n);
    k_2 = x(:,n) + h*A(2,1)*f(k_1);
    k_3 = x(:,n) + h*A(3,1)*f(k_1) + h*A(3,2)*f(k_2);
    k_4 = x(:,n) + h*A(4,1)*f(k_1) + h*A(4,2)*f(k_2) + h*A(4,3)*f(k_3);
    x(:,n+1) = x(:,n) + h*(b(1)*f(k_1) + b(2)*f(k_2) + b(3)*f(k_3) + b(4)*f(k_4));
end

% for n=1:size_t-1
%     x(:,n+1) = x(:,n) + h*f(x(:,n));
% end

plot3(x(1,:),x(2,:),x(3,:));
xlabel("X");
ylabel("Y");
zlabel("Z");

function y_prime = f(y)

sigma = 10;
rho = 28;
beta = 2.7;

y_prime = zeros(3,1);
y_prime(1) = sigma*(y(2)-y(1));
y_prime(2) = y(1)*(rho-y(3))-y(2);
y_prime(3) = y(1)*y(2)-beta*y(3);
end
