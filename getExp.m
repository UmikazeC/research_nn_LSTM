function allData = getExp(dataDir,gt_choices)

     % 获取目录中所有的子文件
    folderList = dir(dataDir);
    folderList = folderList([folderList.isdir] & ~startsWith({folderList.name}, '.'));

    % 提取文件夹名并转换为数字，用于排序
    % 假设文件夹名格式为 't' 后跟数字 (如 t1, t2, ..., t36)
    folderNumbers = str2double(regexprep({folderList.name}, '^t', ''));

    % 对文件夹名进行排序
    [~, sortedIndices] = sort(folderNumbers);
    folderList = folderList(sortedIndices);

    allData = struct('Exp', {});
    idx = 1;
    expidx = 1;
    for folderIdx = 1:length(folderList)
        folderName = folderList(folderIdx).name;
        currentFolder = fullfile(dataDir, folderName);
        files = dir(fullfile(currentFolder, '*.csv'));
        % 读取所有CSV文件
        for fileIdx = 1:length(files)
            file = files(fileIdx);
            fileName = file.name;
              % 解析实验者序号和实验类型
            participantNum = regexp(fileName, '(?<=p_)\d+', 'match', 'once');
            experimentType = regexp(fileName, '(?<=exp_)[A-L]', 'match', 'once');
            
            % 确保participantNum和experimentType都不是空
            if isempty(participantNum) || isempty(experimentType)
                warning('Failed to parse participant number or experiment type from file name: %s', fileName);
                continue;  % 跳过当前文件的处理
            end

            participantField = ['p', participantNum];  % 构造有效的字段名
            filePath = fullfile(currentFolder, file.name);
            csvData = readtable(filePath);

            % find steer time
            manual = find(diff(csvData.is_autonomous) < 0) + 1;
            if ~isempty(manual)
                steer_time = csvData.time_s_(manual(end));
    
                %% find start time
                start = find(diff(csvData.ped0_v_m_s_)>0) + 1;
                start_time = csvData.time_s_(start(1));
    
                %% find reaction time
                rat = steer_time-start_time;

                ped0id = csvData.ped0_val(1);
                ped1id = csvData.ped1_val(1);
                if strcmpi(ped0id, 'Pedophile') || strcmpi(ped0id, 'Rapist') || strcmpi(ped0id, 'Terrorist')
                    ped0val = 0;
                else
                    if strcmpi(ped0id, 'Judge') || strcmpi(ped0id, 'Billionaire') || strcmpi(ped0id, 'Celebrity')
                        ped0val = 1;
                    else
                        ped0val = 2;
                    end
                end
        
                if strcmpi(ped1id, 'Pedophile') || strcmpi(ped1id, 'Rapist') || strcmpi(ped1id, 'Terrorist')
                    ped1val = 0;
                else
                    if strcmpi(ped1id, 'Judge') || strcmpi(ped1id, 'Billionaire') || strcmpi(ped1id, 'Celebrity')
                        ped1val = 1;
                    else
                        ped1val = 2;
                    end
                end
                %% find gaze
                gazex = csvData.gaze_x(start:manual(end));
                gazey = csvData.gaze_y(start:manual(end));
                ped0cx = csvData.ped0_cx(start:manual(end));
                ped0cy = csvData.ped0_cy(start:manual(end));
                ped1cx = csvData.ped1_cx(start:manual(end));
                ped1cy = csvData.ped1_cy(start:manual(end));
                
                ped0dist = sqrt((gazex-ped0cx).*(gazex-ped0cx)+(gazey-ped0cy).*(gazey-ped0cy));
                ped1dist = sqrt((gazex-ped1cx).*(gazex-ped1cx)+(gazey-ped1cy).*(gazey-ped1cy));
                for i = 1:length(gazex)
                    if ped0dist < 0.2
                        ped0focus = 1-5*ped0dist;
                    else
                        ped0focus = 0;
                    end
                end

                for i = 1:length(gazex)
                    if ped1dist < 0.2
                        ped1focus = 1-5*ped1dist;
                    else
                        ped1focus = 0;
                    end
                end
                focusplot = ped0focus-ped1focus;


                %% find start lane
                if ismember(experimentType, {'A', 'B', 'C', 'G', 'H', 'I'})
                    startlane = 1;
                else
                    startlane = -1;
                end
                choice = gt_choices(expidx);

                %% find trail
                %% find steer
                steercommand = csvData.controller_value_theta__turn_Max100_(manual:end);
                brakecommand = csvData.break___(manual:end);
                %% add Exp
                if (rat > 0.2 && rat < 4)
                    newExp = Exp(csvData,focusplot,experimentType,participantField,rat,rat,ped0val,ped1val,startlane,choice,steercommand,brakecommand);
                    allData(idx).Exp = newExp;
                    idx = idx+1;
                end
                
            end
            
            expidx = expidx+1;
            
        end
    end

end