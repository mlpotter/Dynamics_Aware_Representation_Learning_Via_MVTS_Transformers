close all; clear all; clc;

% dx = sigma*(y-x)
% dy = x*(p-z)-y
% dz = x*y - beta*z
t_span = 0:0.01:3.99;
train_trials = 10000;
val_trials = 500;
test_trials = 1000;

train = {}
for T = 1:train_trials
    x = -20 + rand(1,1)*40;%randi([-20,20],1,1);
    y = -20 + rand(1,1)*40;%randi([-20,20],1,1);
    z = 10 + rand(1,1)*30; %randi([10,40],1,1)
    xo = [x;y;z];
    [t,x] = ode45(@(t,x) laODE(t,x),t_span,xo);
    train{T} = reshape(x,[1,size(x,1),size(x,2)]);
end
train = cat(1,train{:});

val = {}
for T = 1:val_trials
    x = -20 + rand(1,1)*40;%randi([-20,20],1,1);
    y = -20 + rand(1,1)*40;%randi([-20,20],1,1);
    z = 10 + rand(1,1)*30; %randi([10,40],1,1)
    xo = [x;y;z];    
    [t,x] = ode45(@(t,x) laODE(t,x),t_span,xo);
    val{T} = reshape(x,[1,size(x,1),size(x,2)]);
end
val = cat(1,val{:});

test = {}
for T = 1:test_trials
    x = -20 + rand(1,1)*40;%randi([-20,20],1,1);
    y = -20 + rand(1,1)*40;%randi([-20,20],1,1);
    z = 10 + rand(1,1)*30; %randi([10,40],1,1)
    xo = [x;y;z];    
    [t,x] = ode45(@(t,x) laODE(t,x),t_span,xo);
    test{T} = reshape(x,[1,size(x,1),size(x,2)]);
end
test = cat(1,test{:});

%%



scatter3(x(:,1),x(:,2),x(:,3),'k')

y = [x(:,2)]'
z = [x(:,3)]'
x = [x(:,1)]'
time = [t]'
hold on
surface([x;x],[y;y],[z;z],[time;time],...
        'facecol','no',...
        'edgecol','interp',...
        'linew',2);
%%
save('train.mat','train')
save('val.mat','val')
save('test.mat','test')

%%
function dxdt = laODE(t,x)
    p = 28; sigma=10; beta = 8/3;

    dxdt = [sigma*(x(2)-x(1));x(1)*(p-x(3))-x(2) ;x(1)*x(2) - beta*x(3)];
    
end

