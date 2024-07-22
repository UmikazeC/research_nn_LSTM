warning off % 关闭报警信息
clc % 清空命令行
clear % 清空变量
close all % 关闭所有图窗
load("traindata.mat")
% 提取特征和标签
numSamples = length(allData);
maxFocusPlotLength = 40;
maxsteerPlotLength = 20;

inputs = [];
targets = [];

for i = 1:numSamples
    expData = allData(i).Exp;
    focusplot_expanded = expData.focusplot(:)';
    padded_focusplot = zeros(1, maxFocusPlotLength);
    padded_focusplot(1:length(focusplot_expanded)) = focusplot_expanded;
    input = [padded_focusplot, expData.reactiontime, expData.ped0val, expData.ped1val, expData.startlane];
    inputs = [inputs; input];

    if length(expData.steer) > maxsteerPlotLength
        steer_expanded = expData.steer(1:maxsteerPlotLength)';
    else
        steer_expanded = expData.steer(:)';
    end
    padded_steer = zeros(1, maxsteerPlotLength);
    padded_steer(1:length(steer_expanded)) = steer_expanded;

    if length(expData.brake) > maxsteerPlotLength
        brake_expanded = expData.brake(1:maxsteerPlotLength)';
    else
        brake_expanded = expData.brake(:)';
    end
    padded_brake = zeros(1, maxsteerPlotLength);
    padded_brake(1:length(brake_expanded)) = brake_expanded;

    % 将所有输出拼接成一个行向量
    target = [padded_steer, padded_brake];
    targets = [targets; target];
end

% 随机打乱数据
randIdx = randperm(numSamples);
inputs = inputs(randIdx, :);
targets = targets(randIdx, :);

% 分割数据为训练集和验证集（80%训练，20%验证）
trainRatio = 0.8;
numTrain = floor(trainRatio * numSamples);

trainInputs = inputs(1:numTrain, :);
trainTargets = targets(1:numTrain, :);

valInputs = inputs(numTrain+1:end, :);
valTargets = targets(numTrain+1:end, :);
valtargetsT = valTargets';
% 归一化
[X_train_1, X_train_input] = mapminmax(trainInputs', -1, 1); % 训练输入归一化
X_test_1 = mapminmax('apply', valInputs', X_train_input); % 测试输入归一化
[Z_train_1, Z_train_output] = mapminmax(trainTargets', -1, 1); % 训练输出归一化
% 设置参数
inputSize = size(X_train_1, 1);
numHiddenUnits = 200; % 增加隐藏单元数
numResponses = size(Z_train_1, 1);

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    dropoutLayer(0.65) % 添加dropout层防止过拟合
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ... % 增加训练迭代次数
    'MiniBatchSize', 128, ... % 调整批量大小
    'GradientThreshold', 1, ...
    'InitialLearnRate', 1e-4, ... % 调整学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 500, ...
    'ValidationData', {X_test_1, valtargetsT}, ... % 设置验证数据
    'ValidationFrequency', 30, ... % 设置验证频率
    'Verbose', 1, ...
    'Plots', 'training-progress');

% 设置LSTM网络并且仿真
[net, info1] = trainNetwork(X_train_1, Z_train_1, layers, options);

% 预测
Y = predict(net, X_test_1);

% 测试结果反归一化
Y_sim = mapminmax('reverse', Y, Z_train_output');

% 分离预测的结果
predicted_steer = Y_sim(1:size(trainTargets, 2)/2, :);
predicted_brake = Y_sim(size(trainTargets, 2)/2 + 1:end, :);

valTargets_steer = valtargetsT(1:size(trainTargets, 2)/2, :);
valTargets_brake = valtargetsT(size(trainTargets, 2)/2 + 1:end, :);

% 计算评价指标

R_steer = corrcoef(valTargets_steer, predicted_steer);
R2_steer = R_steer(1, 2)^2;
MSE_steer = immse(valTargets_steer, double(predicted_steer));
RMSE_steer = sqrt(MSE_steer);

R_brake = corrcoef(valTargets_brake, predicted_brake);
R2_brake = R_brake(1, 2)^2;
MSE_brake = immse(valTargets_brake, double(predicted_brake));
RMSE_brake = sqrt(MSE_brake);

% 显示评价指标
fprintf('Steer - R2=%f, MSE=%f, RMSE=%f\n', R2_steer, MSE_steer, RMSE_steer);
fprintf('Brake - R2=%f, MSE=%f, RMSE=%f\n', R2_brake, MSE_brake, RMSE_brake);
