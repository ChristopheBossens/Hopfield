classdef Hopfield < handle
    %Hopfield - Simulation tool for Hopfield Neural Networks
    %   Detailed explanation goes here
    
    properties
        synapseWeights = [];
        storedPatterns = [];
        
        patternModel = '';
        activityModel = '';
        selfConnections= '';
        
        activityValues = [];
    end
    
    methods
        % Initializes a Hopfield network. 
        % patternModel: 'S' or 'V' for pattern values [-1 1] or [0 1]
        % activityModel: 'S' or 'V' for activity values [-1 1] or [0 1]
        function obj = Hopfield(patternModel, activityModel, selfConnections)
            if nargin == 0
                obj.patternModel = 'S';
                obj.activityModel = 'S';
                obj.selfConnections = 'noself';
            elseif nargin == 2
                obj.patternModel = patternModel;
                obj.activityModel = activityModel;
                obj.selfConnections = 'noself';
            elseif nargin == 3
                obj.patternModel = patternModel;
                obj.activityModel = activityModel;
                obj.selfConnections = selfConnections;
            else
                display('Incorrect number of parameters');
            end
            
            if obj.activityModel =='S'
                obj.activityValues = [-1 1];
            else
                obj.activityValues = [0 1];
            end
        end
        
        % Generates a new pattern
        % patternLength: length of the pattern
        % patternActivty: proportion of active units in the pattern
        function pattern = GeneratePattern(obj,patternLength, patternActivity)
            pattern = -1.*ones(1,patternLength);
            activityIdx = randperm(patternLength);
            pattern(activityIdx(1:round(patternLength*patternActivity))) = -1.*pattern(activityIdx(1:round(patternLength*patternActivity)));
            
            if obj.patternModel == 'V'
                pattern = (pattern + 1)./2;
            end
        end
        
        % Takes the input and flips the state of a randum number of units
        % pattern: the pattern to be changed
        % noiseLevel: proportion of units that will be flipped
        function outputPattern = DistortPattern(obj, inputPattern, noiseLevel)
            flipIndices = randperm(length(inputPattern));
            flipIndices = flipIndices(1:round(noiseLevel*length(inputPattern)));
            
            outputPattern = inputPattern;
            if (obj.patternModel == 'S')
                outputPattern(flipIndices) = -1.*outputPattern(flipIndices);
            elseif (obj.patternModel == 'V')
                outputPattern = (2.*outputPattern)-1;
                outputPattern(flipIndices) = -1.*outputPattern(flipIndices);
                outputPattern = (outputPattern+1)./2;
            end
        end
        
        % Add the pattern nextPattern to the weight matrix
        % C: normalization constant for  the weights, defaults to 1/N
        % A: if specified, will clip abs(synapseWeights) to A
        function AddPattern(obj, nextPattern, C, A)
            if nargin == 2
                C = 1/length(nextPattern);
            end
            
            % Initialize weight matrix if this is the first pattern
            if isempty(obj.synapseWeights)
                obj.synapseWeights = zeros(length(nextPattern));
            else
                if size(obj.synapseWeights) ~= length(nextPattern)
                    error('Pattern size not consistent with current weight matrix');
                end
            end
            
            % Pattern should be Nx1 vector
            if size(nextPattern,1) == 1
                nextPattern = nextPattern';
            end
            
            obj.synapseWeights = obj.synapseWeights + C.*(nextPattern*nextPattern');
            if nargin == 4
                obj.synapseWeights(obj.synapseWeights(:) < -A) = -A;
                obj.synapseWeights(obj.synapseWeights(:) > A) = A;
            end
            
            % Removes self connections
            if strcmp(obj.selfConnections,'noself')
                for i = 1:size(obj.synapseWeights)
                    obj.synapseWeights(i,i) = 0;
                end
            end
            
            % Store the pattern
            obj.storedPatterns = [obj.storedPatterns nextPattern];
        end

        % Resets the weight matrix to zero
        function ResetWeights(obj)
            obj.storedPatterns = [];
            obj.synapseWeights = [];
        end
        
        function result = GetWeightMatrix(obj)
            result = obj.synapseWeights;
        end
        
        % Performs a single update iteration. For asynchronous updating,
        % this corresponds to an update for all the units in random order
        % initialState: initial state of the network
        % method: 'sync' or 'async'
        % threshold: the value to which to compare each units' activity
        function outputState = Iterate(obj, initialState, method,threshold)
            if nargin == 2
                method = 'async';
            elseif nargin == 3
                threshold = 0;
            end
            
            if (size(initialState,1) == 2)
                initialState = initialState';
            end
            
            outputState = initialState;
            switch method
                case 'sync'
                    inputPotential = obj.synapseWeights*initialState';
                    
                    outputState( (inputPotential-threshold) > 0) = obj.activityValues(2);
                    outputState( (inputPotential-threshold) <= 0) = obj.activityValues(1);
                case 'async'
                    updateOrder = randperm(size(obj.synapseWeights,1));
                    for unitIdx = 1:length(updateOrder)
                        inputPotential = obj.synapseWeights(updateOrder(unitIdx),:)*outputState';
                        if ( (inputPotential-threshold) > threshold)
                            outputState(updateOrder(unitIdx)) = obj.activityValues(2);
                        else
                            outputState(updateOrder(unitIdx)) = obj.activityValues(1);
                        end
                    end
            end
        end
        
        function energy = GetEnergy(obj, networkState)
            if (size(networkState,2) > 1)
                networkState = networkState';
            end
            
            energy = -sum(sum(obj.synapseWeights.*(networkState*networkState')));
        end
        
        % Probe the network by starting with different initial random
        % states and keep track of which network states it converges to
        function [stableStates, stateHist] = GetSpuriousStates(obj,nRandomPatterns,p,threshold,activityValues)
            if nargin == 2
                p = 0.5;
                threshold = 0;
                activityValues = [-1 1];
            elseif nargin == 3
                threshold = 0;
                activityValues = [-1 1];
            end
            
            stableStates = zeros(nRandomPatterns,size(obj.synapseWeights,1));
            stateHist = zeros(1,nRandomPatterns);
            nDistinctStates = 0;
            
            for idx = 1:nRandomPatterns
                netInput = activityValues(1).*ones(1,size(obj.synapseWeights,1));
                activityIdx = randperm(size(obj.synapseWeights,1));
                netInput(activityIdx(1:round(p*length(activityIdx)))) = activityValues(2);
                
                converged = -1;
                while converged == -1
                    netOutput = obj.Iterate(netInput,'async',threshold,activityValues);
                    if (sum(netInput-netOutput) == 0)
                        converged = 1;
                    else
                        netInput = netOutput;
                    end
                end
                
                patternIdx = find(ismember(stableStates,netOutput,'rows') == 1);
                if (isempty(patternIdx))
                    patternIdx = find(ismember(stableStates,-1.*netOutput,'rows') == 1);
                    if (isempty(patternIdx))
                        nDistinctStates = nDistinctStates + 1;
                        patternIdx = nDistinctStates;
                    end
                end

                stateHist(patternIdx) = stateHist(patternIdx) + 1;
                stableStates(patternIdx,:) = netOutput;
            end
            stateHist = stateHist(1:nDistinctStates);
            stableStates = stableStates(1:nDistinctStates,:);
        end
    end
    
end

