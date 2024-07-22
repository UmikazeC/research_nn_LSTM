warning off % 关闭报警信息
clc % 清空命令行
clear % 清空变量
close all % 关闭所有图窗
load("traindata.mat")

% 提取特征和标签
numSamples = length(allData);
maxFocusPlotLength = 40;

inputs = [];
targets_choice = [];

for i = 1:numSamples
    expData = allData(i).Exp;
    % 将focusplot展开成固定长度的向量，并填充0值
    focusplot_expanded = expData.focusplot(:)';
    padded_focusplot = zeros(1, maxFocusPlotLength);
    padded_focusplot(1:length(focusplot_expanded)) = focusplot_expanded;
    % 将所有输入特征拼接成一个行向量
    input = [padded_focusplot, expData.reactiontime, expData.ped0val, expData.ped1val, expData.startlane];
    inputs = [inputs; input];
    % 将choice转换为分类标签
    if expData.choice == -1
        targets_choice = [targets_choice; 1];
    elseif expData.choice == 0
        targets_choice = [targets_choice; 2];
    else
        targets_choice = [targets_choice; 3];
    end
end


% 随机打乱数据
randIdx = randperm(numSamples);
inputs = inputs(randIdx, :);
targets_choice = targets_choice(randIdx, :);

% 分割数据为训练集和验证集（80%训练，20%验证）
trainRatio = 0.8;
numTrain = floor(trainRatio * numSamples);

trainInputs = inputs(1:numTrain, :);
trainTargets_choice = targets_choice(1:numTrain, :);

valInputs = inputs(numTrain+1:end, :);
valTargets_choice = targets_choice(numTrain+1:end, :);

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
numHiddenUnits = 200;

layers_classification = [
    sequenceInputLayer(inputSize, 'Name', 'input')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(3, 'Name', 'choice_fc')
    softmaxLayer('Name', 'choice_softmax')
    classificationLayer('Name', 'choice_classification')];

% 创建并训练分类任务的神经网络

options_classification = trainingOptions('adam', ...
    'MaxEpochs',5000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...      
    'LearnRateSchedule','piecewise', ...%每当经过一定数量的时期时，学习率就会乘以一个系数。
    'LearnRateDropPeriod',400, ...      %乘法之间的纪元数由“ LearnRateDropPeriod”控制。可调
    'LearnRateDropFactor',0.15, ...      %乘法因子由参“ LearnRateDropFactor”控制，可调
    'Verbose',0,  ...  %如果将其设置为true，则有关训练进度的信息将被打印到命令窗口中。默认值为true。
    'Plots','training-progress');    %构建曲线图 将'training-progress'替换为none
net_classification = trainNetwork(X_train_Cell, trainTargets_choice_categorical, layers_classification, options_classification);

% 预测分类结果
YPred_classification = classify(net_classification, X_test_Cell);
predicted_choice = double(YPred_classification) - 2; % 将标签转换为-1, 0, 1
gt_choice = valTargets_choice-2;
% 计算分类准确率
accuracy_choice = sum(predicted_choice == gt_choice) / numel(gt_choice);
accuracy_choice_num = sum(predicted_choice == gt_choice);
% 显示分类准确率
fprintf('Choice - Accuracy rate=%f\n', accuracy_choice);
fprintf('Choice - Accuracy num=%f\n', accuracy_choice_num);


