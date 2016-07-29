classdef Hopfield < handle
    %Hopfield - Simulation tool for Hopfield Neural Networks
    %   Detailed explanation goes here
    
    properties
        backupWeights  = [];
        synapseWeights = [];
        storedPatterns = [];
        
        unitModel = 'S';
        selfConnections= 'noself';
        clipMode  = 'noclip';
        clipValue = 1;
        updateMode = 'async';
        
        networkSize = 0;
        unitThreshold = 0;
        unitValues = [-1 1];
        
        maxIterations = 100;
    end
    
    methods
        % Initializes a Hopfield network of a given size
        function obj = Hopfield(networkSize)
            obj.networkSize = networkSize;
            obj.synapseWeights = zeros(obj.networkSize);
        end
        
        % Generates a new pattern
        % patternActivty: proportion of active units in the pattern
        function pattern = GeneratePattern(obj,patternActivity)
            if nargin == 1
                patternActivity = 0.5;
            end
            pattern = -1.*ones(1,obj.networkSize);
            activityIdx = randperm(obj.networkSize);
            pattern(activityIdx(1:round(obj.networkSize*patternActivity))) = -1.*pattern(activityIdx(1:round(obj.networkSize*patternActivity)));
            
            if obj.unitModel == 'V'
                pattern = (pattern + 1)./2;
            end
        end
        
        % Takes the input and flips the state of a randum number of units
        % inputPattern: the pattern to be changed
        % noiseLevel: proportion of units that will be flipped
        function outputPattern = DistortPattern(obj, inputPattern, noiseLevel)
            flipIndices = randperm(length(inputPattern));
            flipIndices = flipIndices(1:round(noiseLevel*length(inputPattern)));
            
            outputPattern = inputPattern;
            if (obj.unitModel == 'S')
                outputPattern(flipIndices) = -1.*outputPattern(flipIndices);
            elseif (obj.unitModel == 'V')
                outputPattern = (2.*outputPattern)-1;
                outputPattern(flipIndices) = -1.*outputPattern(flipIndices);
                outputPattern = (outputPattern+1)./2;
            end
        end
        
        % Add the pattern nextPattern to the weight matrix
        % C: normalization constant for  the weights, defaults to 1/N
        function AddPattern(obj, nextPattern, C)
            if nargin == 2
                C = 1/length(nextPattern);
            end
            
            % Check if pattern has correct length
            if size(obj.synapseWeights) ~= length(nextPattern)
                error('Pattern size not consistent with current weight matrix');
            end
            
            % Pattern should be Nx1 vector
            if size(nextPattern,1) == 1
                nextPattern = nextPattern';
            end
            
            obj.synapseWeights = obj.synapseWeights + C.*(nextPattern*nextPattern');
            
            % Check if clipping needs to be applied
            if strcmp(obj.clipMode,'clip')
                obj.synapseWeights(obj.synapseWeights(:) < -obj.clipValue) = -obj.clipValue;
                obj.synapseWeights(obj.synapseWeights(:) > obj.clipValue) = obj.clipValue;
            end
            
            % Check if self connections need to be removed
            if strcmp(obj.selfConnections,'noself')
                for i = 1:size(obj.synapseWeights)
                    obj.synapseWeights(i,i) = 0;
                end
            end
            
            % Store the pattern. Because low activity patterns need to be
            % modified by the user, we apply a small trick so that the
            % original pattern values are stored
            M = mean(nextPattern);
            nextPattern(nextPattern > M) = obj.unitValues(2);
            nextPattern(nextPattern < M) = obj.unitValues(1);
            obj.storedPatterns = [obj.storedPatterns; nextPattern'];
        end

        % Fetch and manipulate the weight matrix manually
        function ResetWeights(obj)
            obj.storedPatterns = [];
            obj.synapseWeights = zeros(obj.networkSize);
        end   
        function result = GetWeightMatrix(obj)
            result = obj.synapseWeights;
        end
        function SetWeightMatrix(obj, weightMatrix)
            obj.synapseWeights = weightMatrix;
        end
        
        % Configure network parameters
        function SetUnitModel(obj,unitModel)
            if unitModel == 'S'
                obj.unitModel = 'S';
                obj.unitValues = [-1 1];
            elseif unitModel == 'V'
                obj.unitModel = 'V';
                obj.unitValues = [0 1];
            end
        end
        function SetThreshold(obj,T)
            obj.unitThreshold = T;
        end
        function SetUpdateMode(obj,updateMode)
            if updateMode == 'async'
                obj.updateMode ='async';
            elseif updateMode == 'sync';
                obj.updateMode = 'sync';
            else
                display('Invalid update mode. Using default mode (''async'')');
                obj.updateMode = 'async';
            end
        end
        function EnableWeightClipping(obj, clipValue)
            obj.clipMode = 'clip';
            obj.clipValue = clipValue;
        end
        function DisableWeightClipping(obj)
            obj.clipMode = 'noclip';
        end
        function EnableSelfConnections(obj)
            obj.selfConnections = 'self';
        end
        function DisableSelfConnections(obj)
            obj.selfConnections = 'noself';
        end
        
        % Performs a single update iteration. 
        % For synchronous updating, all units are updated in a single
        % operation
        % For asynchronous updating, all units are updated in a random
        % order
        function outputState = Iterate(obj, initialState)
            threshold = obj.unitThreshold;
            method = obj.updateMode;
           
            if (size(initialState,1) == 2)
                initialState = initialState';
            end
            
            outputState = initialState;
            switch method
                case 'sync'
                    inputPotential = obj.synapseWeights*initialState';
                    
                    outputState( (inputPotential-threshold) > threshold) = obj.unitValues(2);
                    outputState( (inputPotential-threshold) <= threshold) = obj.unitValues(1);
                case 'async'
                    updateOrder = randperm(size(obj.synapseWeights,1));
                    for unitIdx = 1:length(updateOrder)
                        inputPotential = obj.synapseWeights(updateOrder(unitIdx),:)*outputState';
                           
                        if ( (inputPotential) > threshold)
                            outputState(updateOrder(unitIdx)) = obj.unitValues(2);
                        else
                            outputState(updateOrder(unitIdx)) = obj.unitValues(1);
                        end
                    end
            end
        end
        
        % Performs multiple iteration steps until the network units no
        % longer change between two succesive updates. The final network
        % state is return together with the number of iterations it took to
        % get there
        function [updatedState,it] = Converge(obj, initialState)
            if (size(initialState,1) == 2)
                initialState = initialState';
            end
            
            it = 0;
            updatedState = initialState;
            initialState = zeros(size(updatedState));
            while sum(initialState ~=  updatedState) > 0
                initialState = updatedState;
                updatedState = obj.Iterate(initialState);
                it = it + 1;
                
                if it > obj.maxIterations
                    display('Warning: maximum number of iterations exceeded');
                    break
                end
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
                randomPattern = obj.GeneratePattern(patternActivity);
                output = obj.Converge(randomPattern);
                
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
                
        function [counts, bins, potentialValues, activityValues] = GetPotentialDistribution(obj, nBins)
            nPatterns = size(obj.storedPatterns,1);
                       
            potentialValues = zeros(1,size(obj.synapseWeights,1)*nPatterns);
            activityValues = zeros(1,size(obj.synapseWeights,1)*nPatterns);
            
            for patternIndex = 1:nPatterns
                nextPattern = obj.storedPatterns(patternIndex,:);
                inputPotential = obj.synapseWeights*nextPattern';                
                potentialValues( (1:size(obj.synapseWeights,1))+((patternIndex-1) * size(obj.synapseWeights,1))) = inputPotential;
                activityValues((1:size(obj.synapseWeights,1))+((patternIndex-1) * size(obj.synapseWeights,1))) = nextPattern;
            end
            
            minBin = min(potentialValues);
            maxBin = max(potentialValues);
            bins = linspace(minBin, maxBin,nBins);
            counts= zeros(2,length(bins));
            counts(1,:) = hist(potentialValues(activityValues > 0),bins);
            counts(2,:) = hist(potentialValues(activityValues <= 0),bins);
        end
    end
    
    
    methods (Static)
        % Takes a weight matrix and a deletion factor d (0 <= d <= 1)
        % d determines, for each neuron, the proportion of incoming
        % synapses that are set to zero
        function prunedWeightMatrix = PruneWeightMatrix(weightMatrix,d)
            prunedWeightMatrix = weightMatrix;

            for rowIndex = 1:size(prunedWeightMatrix,1)
                delIndices = randperm(size(prunedWeightMatrix,2));
                delLimit = round(d*size(prunedWeightMatrix,2));
                if delLimit > 0
                    delIndices = delIndices(1:delLimit);
                    prunedWeightMatrix(rowIndex,delIndices) = 0;
                end
            end
        end
        
        % Takes a trained hopfield network together with a pattern matrix
        % Each pattern in the matrix is distorted with the specified noise
        % level. The disorted pattern is presented to the network and if
        % the network converges to the original pattern, this is considered
        % a correct response. The proportion of correctly recalled patterns
        % is returned
        function [pc, it] = TestPatterns(hopnet, patternMatrix, noiseLevel)
            nPatterns = size(patternMatrix,1);
            
            pcVector = zeros(1,nPatterns);
            itVector = zeros(1,nPatterns);
            
            for patternIndex = 1:nPatterns
                pattern = patternMatrix(patternIndex,:);
                noisePattern = hopnet.DistortPattern(pattern, noiseLevel);
                
                [finalState, nIterations] = hopnet.Converge(noisePattern);
                itVector(patternIndex) = nIterations;
                
                if sum(pattern == finalState) == size(patternMatrix,2)
                    pcVector(patternIndex) = 1;
                end
            end
            
            pc = mean(pcVector);
            it = mean(itVector);
        end
    end
end

