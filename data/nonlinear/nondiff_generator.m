close all; clear all; clc;

% dx = sigma*(y-x)
% dy = x*(p-z)-y
% dz = x*y - beta*z
t_span = 0:0.02:10.99;
train_trials = 1;
val_trials = 1;
test_trials = 1;

train = {}
for T = 1:train_trials
    x = -5+ rand(1,1)*10;
    y = -5 + rand(1,1)*10;
    xo = [x;y];
    [t,x] = ode45(@(t,x) nondiff(t,x),t_span,xo);
    train{T} = reshape(x,[1,size(x,1),size(x,2)]);
end
train = cat(1,train{:});

val = {}
for T = 1:val_trials
    x = -5 + rand(1,1)*10;
    y = -5 + rand(1,1)*10;
    xo = [x;y];
    [t,x] = ode45(@(t,x) nondiff(t,x),t_span,xo);
    val{T} = reshape(x,[1,size(x,1),size(x,2)]);
end
val = cat(1,val{:});

test = {}
for T = 1:test_trials
    x = -5 + rand(1,1)*10;
    y = -5 + rand(1,1)*10;
    xo = [x;y] ;  
    [t,x] = ode45(@(t,x) nondiff(t,x),t_span,xo);
    test{T} = reshape(x,[1,size(x,1),size(x,2)]);
end
test = cat(1,test{:});
%%
figure
j = 1
plot(train(j,:,1))
hold on
plot(train(j,:,2))
legend(["x_1","x_2"])
xlabel("Time Steps")
title("States")
figure
plot(train(j,:,1),train(j,:,2))
%% 3D model version...
mu = -0.2; lambda=-.8;
y0 = [xo(1) ;xo(2) ;xo(1)^2]
% y0 = [.1; .2 ; .3]
Y = y0
y = y0
A = [mu 0 0; 0 lambda -lambda;  0 0 2*mu];
for i = t_span
    y = A*y;
    Y = [Y y];
end
figure
plot([t_span 21],Y(1,:))
hold on
plot([t_span 21],Y(2,:))
plot([t_span 21],Y(3,:))

% figure 
% scatter3(Y(1,:),Y(2,:),Y(3,:))
%%
%%
save('C:\Users\lpott\Desktop\DYAN\Code\data\nonlinear\train.mat','train')
save('C:\Users\lpott\Desktop\DYAN\Code\data\nonlinear\val.mat','val')
save('C:\Users\lpott\Desktop\DYAN\Code\data\nonlinear\test.mat','test')
%%




function dxdt = nondiff(t,x)
    mu = -0.2; lambda=-.8;

    dxdt = [mu*x(1) ; lambda * (x(2)-x(1)^2)];
end

