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
            if nargin < 3
                selfConnections = 'noself';
            end
            if nargin < 2
                activityModel = 'S';
            end
            if nargin < 1
                patternModel = 'S';
            end
            
            if ~(strcmp(activityModel,'V') || strcmp(activityModel,'S'))
                display('Unrecognized activity model. Resorting to default');
                activityModel = 'S';
            end

           if ~(strcmp(patternModel,'V') || strcmp(patternModel,'S'))
                display('Unrecognized pattern model. Resorting to default');
                patternModel = 'S';
           end
            
            obj.patternModel = patternModel;
            obj.activityModel = activityModel;
            obj.selfConnections = selfConnections;
            
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
            if nargin == 2
                patternActivity = 0.5;
            end
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
            
            % Store the pattern. Because low activity patterns need to be
            % modified by the user, we apply a small trick so that the
            % original pattern values are stored
            M = mean(nextPattern);
            nextPattern(nextPattern > M) = obj.activityValues(2);
            nextPattern(nextPattern < M) = obj.activityValues(1);
            obj.storedPatterns = [obj.storedPatterns; nextPattern'];
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
                threshold = 0;
            elseif nargin == 3
                threshold = 0;
            end
            
            if (size(initialState,1) == 2)
                initialState = initialState';
            end
            
            effectiveThreshold = threshold;
            outputState = initialState;
            switch method
                case 'sync'
                    inputPotential = obj.synapseWeights*initialState';
                    
                    outputState( (inputPotential-threshold) > threshold) = obj.activityValues(2);
                    outputState( (inputPotential-threshold) <= threshold) = obj.activityValues(1);
                case 'async'
                    updateOrder = randperm(size(obj.synapseWeights,1));
                    for unitIdx = 1:length(updateOrder)
                        inputPotential = obj.synapseWeights(updateOrder(unitIdx),:)*outputState';
                           
                        if ( (inputPotential-effectiveThreshold) > threshold)
                            outputState(updateOrder(unitIdx)) = obj.activityValues(2);
                        else
                            outputState(updateOrder(unitIdx)) = obj.activityValues(1);
                        end
                    end
            end
        end
        
        % Performs multiple iteration steps until the network units no
        % longer change between two succesive updates. The final network
        % state is return together with the number of iterations it took to
        % get there
        function [updatedState,it] = Converge(obj, initialState, method, threshold)
            if nargin == 2
                method = 'async';
                threshold = 0;
            elseif nargin == 3
                threshold = 0;
            end
            
            if (size(initialState,1) == 2)
                initialState = initialState';
            end
            
            it = 0;
            updatedState = initialState;
            initialState = zeros(size(updatedState));
            while sum(initialState ~=  updatedState) > 0
                initialState = updatedState;
                updatedState = obj.Iterate(initialState,method,threshold);
                it = it + 1;
            end
        end
        
        % Returns the energy associated with a specific network state
        function energy = GetEnergy(obj, networkState)
            if (size(networkState,2) > 1)
                networkState = networkState';
            end
            
            energy = -sum(sum(obj.synapseWeights.*(networkState*networkState')));
        end
        
        % Probe the network for the existance of spurious states. For this,
        % a number of random patterns are generated with specified
        % probability of activation. The network is presented with each of
        % these patterns and allowed to settle into a stable state. The
        % number of different states together with their final values is
        % recorded
        function [stableStates, stateHist] = GetSpuriousStates(obj,nRandomPatterns,patternActivity)
            if nargin == 2
                patternActivity = 0.5;
            end
            
            stableStates = zeros(nRandomPatterns,size(obj.synapseWeights,1));
            stateHist = zeros(1,nRandomPatterns);
            nDistinctStates = 0;
            
            for idx = 1:nRandomPatterns
                randomPattern = obj.GeneratePattern(size(obj.synapseWeights,1),patternActivity);
                output = obj.Converge(randomPattern,'async');
                
                patternIdx = find(ismember(stableStates,output,'rows') == 1);
                if (isempty(patternIdx))
                    patternIdx = find(ismember(stableStates,-1.*output,'rows') == 1);
                    if (isempty(patternIdx))
                        nDistinctStates = nDistinctStates + 1;
                        patternIdx = nDistinctStates;
                    end
                end

                stateHist(patternIdx) = stateHist(patternIdx) + 1;
                stableStates(patternIdx,:) = output;
            end
            stateHist = stateHist(1:nDistinctStates);
            stableStates = stableStates(1:nDistinctStates,:);
        end
        
        % Returns the postsynaptic potential values for a given neuron
        function postsynapticPotentials = GetInputPotential(obj, currentState, neuronIndex)
            postsynapticPotentials = sum(obj.synapseWeights(neuronIndex,:).*currentState);
        end
        
        % Returns the input potential for active and inactive neurons in
        % the given state
        function [activePotentials,inactivePotentials] = GetActivityPotentials(obj,currentState)
            if size(currentState,1) < size(currentState,2)
                currentState = currentState';
            end
            
            activeUnits = find(currentState == obj.activityValues(2));
            inactiveUnits = find(currentState == obj.activityValues(1));
            
            activePotentials = obj.synapseWeights(activeUnits,activeUnits)*currentState(activeUnits);
            inactivePotentials = obj.synapseWeights(inactiveUnits,inactiveUnits)*currentState(inactiveUnits);
        end
        
        % Returns the distribution of active and inactive neurons for all
        % patterns in stored in the network
        function [activeDistribution, inactiveDistribution,binValues] = GetActivityDistribution(obj,nBins)
            if nargin == 1
                nBins = 100;
            end
            
            nPatterns = size(obj.storedPatterns,1);
            activeUnitPotentials = [];
            inactiveUnitPotentials = [];
            for patternIndex = 1:nPatterns
                [aup, iup] = obj.GetActivityPotentials(obj.storedPatterns(patternIndex,:));
                activeUnitPotentials = [activeUnitPotentials aup];
                inactiveUnitPotentials = [inactiveUnitPotentials iup];
            end
            
            maxBin = max([activeUnitPotentials(:); inactiveUnitPotentials(:)]);
            minBin = min([activeUnitPotentials(:); inactiveUnitPotentials(:)]);
            binValues = linspace(minBin,maxBin,nBins);
            
            activeDistribution = hist(activeUnitPotentials(:),binValues);
            inactiveDistribution = hist(inactiveUnitPotentials(:),binValues);
            
        end
    end
    
end

