clc
clear
close all
warning('off')

gt_choices = importdata('ground_truth_choices.txt');

dataDir = 'D:\Research\csv_ana\csv';
allData = getExp(dataDir,gt_choices);

save("traindata.mat","allData")