clc
clear
close all
warning('off')

load("traindata.mat")

% 提取特征和标签
numSamples = length(allData);
maxFocusPlotLength = 60;

inputs = [];
targets_choice = [];
startlane = [];

for i = 1:numSamples
    expData = allData(i).Exp;
    % 将focusplot展开成固定长度的向量，并填充0值
    focusplot_expanded = expData.focusplot(:)';
    padded_focusplot = zeros(1, maxFocusPlotLength);
    padded_focusplot(1:length(focusplot_expanded)) = focusplot_expanded;

    ped0dist_expanded = expData.ped0d(:)';
    padded_ped0dist = zeros(1, maxFocusPlotLength);
    padded_ped0dist(1:length(ped0dist_expanded)) = ped0dist_expanded;

    ped1dist_expanded = expData.ped1d(:)';
    padded_ped1dist = zeros(1, maxFocusPlotLength);
    padded_ped1dist(1:length(ped1dist_expanded)) = ped1dist_expanded;
    % 将所有输入特征拼接成一个行向量
    input = [expData.reactiontime, expData.ped0val, expData.ped1val, expData.startlane, padded_focusplot, padded_ped0dist, padded_ped1dist];
    startlane = [startlane;expData.startlane];
    inputs = [inputs; input];
    % 将choice转换为分类标签
    if expData.choice == -1
        targets_choice = [targets_choice; 1];
    else
        targets_choice = [targets_choice; 2];
    end
end


% 随机打乱数据
randIdx = randperm(numSamples);
inputs = inputs(randIdx, :);
targets_choice = targets_choice(randIdx, :);
targets_startlane = startlane(randIdx, :);

% 分割数据为训练集和验证集（80%训练，20%验证）
trainRatio = 0.8;
numTrain = floor(trainRatio * numSamples);

trainInputs = inputs(1:numTrain, :);
trainTargets_choice = targets_choice(1:numTrain, :);

valInputs = inputs(numTrain+1:end, :);
valTargets_choice = targets_choice(numTrain+1:end, :);
valtargets_startlane = targets_startlane(numTrain+1:end, :);
% 将分类标签转换为分类数组
trainTargets_choice_categorical = categorical(trainTargets_choice);
valTargets_choice_categorical = categorical(valTargets_choice);

% 归一化
[X_train, X_train_input] = mapminmax(trainInputs', -1, 1); % 训练输入归一化
X_test = mapminmax('apply', valInputs', X_train_input); % 测试输入归一化

% 将输入和输出转换为单元数组
X_train_Cell = num2cell(X_train, 1)';
X_test_Cell = num2cell(X_test, 1)';

% 设置参数
inputSize = size(X_train, 1);
numHiddenUnits = 100; % 减少LSTM层的单元数
dropoutRate = 0.6; % 增加dropout比例
globalL2RegFactor = 0.1; % 全局L2正则化系数
layerL2RegFactor = 0.1; % 层特定的L2正则化因子

layers_classification = [
    sequenceInputLayer(inputSize, 'Name', 'input')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1')
    dropoutLayer(dropoutRate, 'Name', 'dropout')
    fullyConnectedLayer(2, 'Name', 'choice_fc', 'WeightL2Factor', layerL2RegFactor, 'BiasL2Factor', layerL2RegFactor) % 使用L2正则化因子
    softmaxLayer('Name', 'choice_softmax')
    classificationLayer('Name', 'choice_classification')];

% 创建并训练分类任务的神经网络
options_classification = trainingOptions('adam', ...
    'MaxEpochs', 1000, ... % 增加训练迭代次数
    'MiniBatchSize', 64, ... % 调整批量大小
    'InitialLearnRate', 1e-4, ... % 减少初始学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 250, ...
    'L2Regularization', globalL2RegFactor, ... % 全局L2正则化系数
    'ValidationData', {X_test_Cell, valTargets_choice_categorical}, ... % 设置验证数据
    'ValidationFrequency', 30, ... % 设置验证频率
    'Verbose', 1, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ... % 每个epoch打乱数据
    'ExecutionEnvironment', 'auto');

net_classification = trainNetwork(X_train_Cell, trainTargets_choice_categorical, layers_classification, options_classification);



% 预测分类结果
YPred_classification = classify(net_classification, X_test_Cell);
predicted_choice = double(YPred_classification)*2 - 3; % 将标签转换为-1,  1
gt_choice = valTargets_choice*2-3;
% 计算分类准确率
accuracy_choice = sum(predicted_choice == gt_choice) / numel(gt_choice);
accuracy_choice_num = sum(predicted_choice == gt_choice);
% 显示分类准确率
fprintf('Choice - Accuracy rate=%f\n', accuracy_choice);
fprintf('Choice - Accuracy num=%f\n', accuracy_choice_num);


