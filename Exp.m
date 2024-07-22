classdef Exp
    properties
        alldata
        focusplot
        type
        participant
        reactiontime
        decisiontime
        ped0val
        ped1val
        startlane
        choice
        steer
        brake
        
    end
    methods
        function obj = Exp(alldata,focusplot,type, participant,reactiontime,decisiontime,ped0val,ped1val,startlane,choice,steer,brake)
            obj.alldata = alldata;
            obj.focusplot = focusplot;
            obj.type = type;
            obj.participant = participant;   
            obj.reactiontime = reactiontime;
            obj.decisiontime = decisiontime;
            obj.ped0val = ped0val;
            obj.ped1val = ped1val;
            obj.startlane = startlane;
            obj.choice = choice;
            obj.steer = steer;
            obj.brake = brake;
        end
        
    end
end