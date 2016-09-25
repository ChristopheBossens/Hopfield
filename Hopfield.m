classdef Hopfield < handle
    %Hopfield - Simulation tool for Hopfield Neural Networks
    %   Detailed explanation goes here
    
    properties
        backupWeights  = [];
        synapseWeights = [];
        storedPatterns = [];
        currentState   = [];
        
        unitModel = 'S';
        selfConnections= 'noself';
        clipMode  = 'noclip';
        clipValue = 1;
        updateMode = 'async';
        activityNormalization = 'on';
        updateDynamics = 'deterministic';
        beta = 1.0;
        
        gamma = 1.0;
        networkSize = 0;
        unitThreshold = 0;
        unitValues = [-1 1];
        
        maxIterations = 100;
        maxDeltaIterationMultiplier = 100;
    end
    
    methods
        % Initializes a Hopfield network of a given size
        function obj = Hopfield(networkSize)
            obj.networkSize    = networkSize;
            obj.synapseWeights = zeros(obj.networkSize);
            obj.unitThreshold  = zeros(1,obj.networkSize);
            obj.currentState   = zeros(1,obj.networkSize);
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
        function patternMatrix = GeneratePatternMatrix(obj, nPatterns, patternActivity)
            if nargin == 2
                patternActivity = 0.5;
            end
            patternMatrix = zeros(nPatterns, obj.networkSize);
            
            for patternIndex = 1:nPatterns
                isValidPattern = 0;
                while isValidPattern == 0
                   newPattern = obj.GeneratePattern(patternActivity);

                   isValidPattern = 1;
                   for i = 1:patternIndex
                      if sum(newPattern == patternMatrix(i,:)) == obj.networkSize
                          isValidPattern = 0;
                          break;
                      end
                   end
                end
                patternMatrix(patternIndex,:) = newPattern;
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
            if obj.networkSize ~= length(nextPattern)
                error('Pattern size not consistent with current weight matrix');
            end
            
            % Pattern should be Nx1 vector
            if size(nextPattern,1) == 1
                nextPattern = nextPattern';
            end
            
            if strcmp(obj.activityNormalization,'on')
                nextPattern = nextPattern-mean(nextPattern);
            end
            
            obj.synapseWeights = obj.gamma.*obj.synapseWeights + C.*(nextPattern*nextPattern');
            
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
            
            % Store the pattern
            obj.storedPatterns = [obj.storedPatterns; nextPattern'];
        end
        function AddPatternMatrix(obj,patternMatrix,C)
            if nargin == 2
                C = 1/size(patternMatrix,2);
            end
            
            for i = 1:size(patternMatrix,1)
                obj.AddPattern(patternMatrix(i,:),C);
            end
        end
        
        % Add a pattern to the matrix using the delta learning rule
        function outputError = PerformDeltaUpdate(obj, nextPattern, etha)
            if obj.networkSize ~= length(nextPattern)
                error('Pattern size not consistent with current weight matrix');
            end
            
            currentOutput = obj.Iterate(nextPattern);
            outputError = nextPattern(:) - currentOutput(:);           
            deltaW = outputError*nextPattern(:)' + nextPattern(:)*outputError(:)';
            obj.synapseWeights = obj.synapseWeights + etha.*deltaW;
            
            if strcmp(obj.selfConnections,'noself')
                obj.synapseWeights(1:(obj.networkSize+1):(obj.networkSize*obj.networkSize)) = 0;
            end
        end
        
        % Add patterns from the input matrix using the delta learning rule
        function LearnDeltaPatterns(obj, patternMatrix, etha)
            zeroError = 0;
            currentIteration = 1;
            nPatterns = size(patternMatrix,1);
            
            while zeroError == 0
                err = zeros(1,nPatterns);
                for trainingIndex = 1:nPatterns
                    err(trainingIndex) = mean(abs(obj.PerformDeltaUpdate(patternMatrix(trainingIndex,:),etha)));
                end
                
                if sum(err) == 0
                    zeroError = 1;
                end
                
                currentIteration = currentIteration + 1;
                if currentIteration > (obj.maxDeltaIterationMultiplier*nPatterns)
                    error('Maximum training iterations for delta learning rule exceeded!');
                end
            end
        end
        % Fetch and manipulate the weight matrix manually
        function ResetWeightMatrix(obj)
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
        function SetGamma(obj,g)
            obj.gamma = g;
        end
        function SetThreshold(obj,T)
            if length(T) == 1
                obj.unitThreshold = T.*ones(1,obj.networkSize);
            elseif length(T) == obj.networkSize
                obj.unitThreshold = T;
            else
                error('Length of T needs to be equal to one or to the number of units in the network');
            end
        end
        function SetUpdateMode(obj,updateMode)
            if strcmp(updateMode,'async')==1
                obj.updateMode ='async';
            elseif strcmp(updateMode,'sync')==1
                obj.updateMode = 'sync';
            else
                display('Invalid update mode. Using default mode (''async'')');
                obj.updateMode = 'async';
            end
        end
        function SetActivityNormalization(obj, activityNormalization)
            activityNormalization = lower(activityNormalization);
            validModes = {'on','off'};
            
            for i = 1:length(validModes)
                if strcmpi(activityNormalization,validModes{i})
                    obj.activityNormalization = activityNormalization;
                    return;
                end
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
        function UseDeterministicDynamics(obj)
            obj.updateDynamics = 'deterministic';
        end
        function UseStochasticDynamics(obj,beta)
            obj.updateDynamics = 'stochastic';
            obj.beta = beta;
        end
        function currentState = GetCurrentState(obj)
            currentState = obj.currentState;
        end
        % Clamp the state of the model to a specific set of values
        function ClampState(obj,newState)
            if length(newState) ~= obj.networkSize
                error('Number of units does not correspond')
            end
            if size(newState,2) == 1
                newState = newState';
            end
            obj.currentState = newState;
        end
        
        % Here we perform the state update of the hopfield network. Updates
        % are performed synchronously (i.e. all units at the same time), or
        % asynchronously (each unit in turn, but in random order)
        function output = Iterate(obj, initialState)
            threshold = obj.unitThreshold;
            method = obj.updateMode;      
            obj.ClampState(initialState);
            
            switch method
                case 'sync'
                    inputPotential = obj.synapseWeights*obj.currentState';
                    
                    if strcmp(obj.updateDynamics,'deterministic') == 1
                        obj.currentState( (inputPotential) > threshold) = obj.unitValues(2);
                        obj.currentState( (inputPotential) <= threshold) = obj.unitValues(1);
                    else
                        g = 0.5.*(1+tanh(obj.beta.*(inputPotential-threshold)));
                        r = rand(size(g));
                        obj.currentState(g > r) = obj.unitValues(2);
                        obj.currentState(g <= r) = obj.unitValues(1);
                    end
                case 'async'
                    updateOrder = randperm(size(obj.synapseWeights,1));
                    for unitIdx = 1:length(updateOrder)
                        obj.UpdateUnit(updateOrder(unitIdx));
                    end
            end
            output = obj.currentState;
        end
        
        % Updates the state of a single unit, given the current network
        % state
        function output = UpdateUnit(obj, unitIndex)
            if length(unitIndex(:)) ~= 1
                error('Only provide a single unit index')
            end
            
            inputPotential = obj.synapseWeights(unitIndex,:)*obj.currentState';
            if strcmp(obj.updateDynamics,'deterministic') == 1
                if ( (inputPotential) > obj.unitThreshold(unitIndex))
                    obj.currentState(unitIndex) = obj.unitValues(2);
                else
                    obj.currentState(unitIndex) = obj.unitValues(1);
                end
            else
                g = 0.5*(1+tanh(obj.beta*(inputPotential - obj.unitThreshold(unitIndex))));
                r = rand();
                if (g > r)
                    obj.currentState(unitIndex) = obj.unitValues(2);
                else
                    obj.currentState(unitIndex) = obj.unitValues(1);
                end
            end
            output = obj.currentState;
        end
        
        % Performs multiple iteration steps until the network units no
        % longer change between two succesive updates. The final network
        % state is return together with the number of iterations it took to
        % get there
        function [finalState,it] = Converge(obj, initialState)
            if (size(initialState,2) == 1)
                initialState = initialState';
            end
            
            it = 0;
            finalState = initialState;
            initialState = 0.*initialState;
            while sum(initialState ~=  finalState) > 0
                if it > obj.maxIterations
                    display('Warning: maximum number of iterations exceeded');
                    break
                end
                
                initialState = finalState;
                finalState = obj.Iterate(initialState);
                it = it + 1;
            end
        end
        
        % Returns the energy associated with a specific network state. The
        % function does not take into account if the given state is stable
        % or not.
        function energy = GetEnergy(obj, networkState)
            if (size(networkState,2) > 1)
                networkState = networkState';
            end
            
            energy = -sum(sum(obj.synapseWeights.*(networkState*networkState')));
        end
        
        % Returns the energy associated with each unit in the following
        % state. The function does not take into account if the given state
        % is stable or not. Unit energy is defined as the activity of the
        % unit multiplied by its net input
        function unitEnergy = GetUnitEnergy(obj,networkState)
            if size(networkState,1) == 1
                networkState = networkState';
            end
            
            netInput = obj.synapseWeights*networkState;
            unitEnergy = netInput.*networkState;
        end
        
        % Generates a random pattern and lets the network converge to a
        % stable state. All weights are then adapted using an unlearning
        % rule
        function Unlearn(obj, epsilon, patternActivity)
            if nargin == 2
                patternActivity = 0.5;
            end
            
            randomPattern = obj.GeneratePattern(patternActivity);
            stableState = obj.Converge(randomPattern);
            
            if size(stableState,1) == 1
                stableState = stableState';
            end
            
            deltaW = -epsilon.*(stableState*stableState');
            obj.synapseWeights = obj.synapseWeights + deltaW;
            for i = 1:size(obj.synapseWeights,1)
                obj.synapseWeights(i,i) = 0;
            end
        end
        
        % Probe the network for the existance of spurious states. For this,
        % a number of random patterns are generated with specified
        % probability of activation. The network is presented with each of
        % these patterns and allowed to settle into a stable state. The
        % number of different states together with their final values is
        % recorded
        function [stablePatternMatrix, stablePatternCount, meanPatternIterations] = SampleWithRandomStates(obj,nRandomPatterns,patternActivity)
            if nargin == 2
                patternActivity = 0.5;
            end
            
            nStableStates = 0;
            stablePatternMatrix = [];
            stablePatternCount  = [];
            meanPatternIterations = [];
            
            for i = 1:nRandomPatterns
                testPattern = obj.GeneratePattern(patternActivity);
                [finalState,it] = obj.Converge(testPattern);
                
                isNewState = 1;
                for j = 1:nStableStates
                    if (sum(stablePatternMatrix(j,:) == finalState) == obj.networkSize) || ...
                        (sum(stablePatternMatrix(j,:) == -finalState) == obj.networkSize)
                        
                        stablePatternCount(j) = stablePatternCount(j) + 1;
                        meanPatternIterations(j) = meanPatternIterations(j) + it;
                        isNewState = 0;
                        break;
                    end
                end
            
                if isNewState == 1
                    stablePatternMatrix = [stablePatternMatrix; finalState];
                    stablePatternCount = [stablePatternCount 1];
                    meanPatternIterations= [meanPatternIterations it];
                    nStableStates = nStableStates + 1;
                end
            end
            meanPatternIterations = meanPatternIterations./stablePatternCount;
        end
        
        % Takes each pattern in the stored pattern matrix in constructs the
        % distribution of inputs for each active and inactive unit in each
        % pattern.
        function [counts, bins, potentialValues, activityValues] = GetPotentialDistribution(obj, nBins)
            if nargin == 1
                nBins = 100;
            end
            
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
        
        % Adds each pattern in pattern matrix to the network. 
        % After each new pattern is added, recall is tested for all
        % previous patterns learned up to that pattern.
        function [overlapVector, pcVector, itVector] = ProbeCapacity(obj,patternMatrix,noiseLevel,C)
            nPatterns = size(patternMatrix,1);
            overlapVector = zeros(1,nPatterns);
            pcVector = zeros(1,nPatterns);
            itVector = zeros(1,nPatterns);
            
            obj.ResetWeights();
            for patternIndex = 1:nPatterns
                obj.AddPattern(patternMatrix(patternIndex,:),C);
                
                for testIndex = 1:patternIndex
                    originalPattern = patternMatrix(patternIndex,:);
                    noisePattern = obj.DistortPattern(originalPattern, noiseLevel);
                    [finalState, nIterations] = obj.Converge(noisePattern);
                    
                    itVector(patternIndex) = itVector(patternIndex) + nIterations;
                    if sum(originalPattern == finalState) == size(patternMatrix,2)
                        pcVector(patternIndex) = pcVector(patternIndex) + 1;
                    end
                    overlapVector(patternIndex) = overlapVector(patternIndex) + (originalPattern*finalState')/obj.networkSize;
                end
                
                overlapVector(patternIndex) = overlapVector(patternIndex)./patternIndex;
                itVector(patternIndex) = itVector(patternIndex)./patternIndex;
                pcVector(patternIndex) = pcVector(patternIndex)./patternIndex;
            end
        end
        
        function [overlapVector, itVector] = TestRecall(obj,patternMatrix,noiseLevel)
            nPatterns = size(patternMatrix,1);
            
            overlapVector = zeros(1,nPatterns);
            itVector = zeros(1,nPatterns);
            
            for patternIndex = 1:nPatterns
                originalPattern = patternMatrix(patternIndex,:);
                distortedPattern = obj.DistortPattern(originalPattern,noiseLevel);
                [finalState,it] = obj.Converge(distortedPattern);
                
                overlapVector(patternIndex) = (finalState*originalPattern')/obj.networkSize;
                itVector(patternIndex) = it;
            end
        end
        
        function isStable = IsStablePattern(obj,testPattern)
            output = obj.Iterate(testPattern);
            
            if sum(output==testPattern) == obj.networkSize
                isStable = 1;
            else
                isStable = 0;
            end
        end
    end
    
    
    methods (Static)
        % Takes a weight matrix and a deletion factor d (0 <= d <= 1)
        % Depending on the specified method, the following happens:
        % - 'incoming': for each neuron a proportion of the incoming
        % connections are deleted
        % - 'random': A random proportion of synapses is removed
        % - 'neuron': A random proportion of neurons is removed
        function [prunedWeightMatrix, deletedWeights, remainingIndices, remainingNeurons] = PruneWeightMatrix(weightMatrix,d, method)
            if nargin == 2
                method = 'synapse';
            end
            
            networkSize = size(weightMatrix,2);
            prunedWeightMatrix = weightMatrix;           
            selfIndices = 1:(networkSize+1):length(prunedWeightMatrix(:));
            remainingNeurons = [];
            
            switch method
                case 'synapse'
                    delLimit = round(d*networkSize);
                                       
                    if delLimit == 0
                        deletedWeights = 0;
                        remainingIndices = setdiff(1:(networkSize*networkSize),selfIndices);
                    else
                        deletedWeights = zeros(networkSize,delLimit);
                        nRemainingIndices = networkSize-delLimit-1;
                        remainingIndices = zeros(networkSize,nRemainingIndices);
                    
                        for rowIndex = 1:size(prunedWeightMatrix,1)
                            delIndices = setdiff(randperm(networkSize),rowIndex);
                            delIndices = delIndices(randperm(length(delIndices)));

                            deletedWeights( rowIndex,:) = prunedWeightMatrix(rowIndex,delIndices(1:delLimit))';
                            prunedWeightMatrix(rowIndex,delIndices(1:delLimit)) = 0;
                            remainingIndices(rowIndex,:) = sub2ind(size(weightMatrix),rowIndex.*ones(1,nRemainingIndices),delIndices( (delLimit+1):end));
                        end
                    end
                    remainingIndices = remainingIndices(:);
                case 'random'
                    delIndices = randperm(length(prunedWeightMatrix(:)));
                    delIndices = setdiff(delIndices,selfIndices);
                    delIndices = delIndices(randperm(length(delIndices)));
                    
                    delLimit = round(d*length(prunedWeightMatrix(:)));
                    
                    if delLimit == 0
                        deletedWeights = 0;
                        remainingIndices = setdiff(1:(networkSize*networkSize),selfIndices);
                    else 
                        deletedWeights = prunedWeightMatrix(delIndices(1:delLimit));
                        remainingIndices = setdiff(1:(networkSize*networkSize),[delIndices(1:delLimit) selfIndices]);
                        prunedWeightMatrix(delIndices(1:delLimit)) = 0;
                        remainingIndices = setdiff(remainingIndices(:),selfIndices);
                    end
                    
                case 'neuron'
                    delNeurons = randperm(networkSize);
                    delLimit = round(d*networkSize);
                    
                    if delLimit == 0
                        deletedWeights = 0;
                        remainingIndices = 1:(networkSize*networkSize);
                        remainingNeurons = 1:networkSize;
                    else
                        deletedWeights = prunedWeightMatrix(:,delNeurons(1:delLimit));
                        prunedWeightMatrix(:,delNeurons(1:delLimit)) = 0;
                        deletedWeights = [deletedWeights; prunedWeightMatrix(delNeurons(1:delLimit),:)'];
                        prunedWeightMatrix(delNeurons(1:delLimit),:) = 0;
                        
                        deletedWeights = deletedWeights(:)';
                        
                        remainingNeurons = delNeurons((delLimit+1):end);
                        nRemainingNeurons = length(remainingNeurons);
                        remainingIndices = zeros(2*nRemainingNeurons,nRemainingNeurons);
                        for j = 1:nRemainingNeurons
                            remainingIndices( 2*j - 1,:) = sub2ind(size(prunedWeightMatrix),remainingNeurons,remainingNeurons(j).*ones(1,nRemainingNeurons));
                            remainingIndices( 2*j, :)    = sub2ind(size(prunedWeightMatrix),remainingNeurons(j).*ones(1,nRemainingNeurons),remainingNeurons);
                        end
                        selfIndices = 1:(networkSize+1):length(prunedWeightMatrix(:));
                        remainingIndices = unique(setdiff(remainingIndices(:),selfIndices));
                    end
            end           
        end
        
        % Takes a trained hopfield network together with a pattern matrix
        % Each pattern in the matrix is distorted with the specified noise
        % level. The disorted pattern is presented to the network and if
        % the network converges to the original pattern, this is considered
        % a correct response. For each sample, a vector contains 1 and 0
        % for each pattern that is succesfully recalled or not.
        function [pcVector, itVector] = TestPatterns(hopnet, patternMatrix, noiseLevel)
            if nargin == 2
                noiseLevel = 0;
            end
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
        end
        
        % This function can be used to test if rows in stablePatternMatrix
        % are present in patternMatrix. The function tests for original and
        % inverse patterns. You will normally use this function after
        % sampling a network with random states to see which of the
        % converged states correspond to stored memories versus spurious
        % states
        function [isSpuriousState, oldPatternIndex] = AnalyseStableStates(stablePatternMatrix,patternMatrix)
            nStableStates = size(stablePatternMatrix,1);
            nPatterns = size(patternMatrix,1);
            networkSize = size(patternMatrix,2);
            oldPatternIndex = -1.*ones(1,nStableStates);
            isSpuriousState = ones(1,nStableStates);
            for j = 1:nStableStates
                for i = 1:nPatterns
                    if (sum(stablePatternMatrix(j,:) == patternMatrix(i,:)) == networkSize) || ...
                            (sum(stablePatternMatrix(j,:) == -patternMatrix(i,:)) == networkSize)
                        isSpuriousState(j) = 0;
                        oldPatternIndex(j) = i;
                        break;
                    end
                end
            end
        end
        
    end
end

