classdef Hopfield < handle
    %Hopfield - Simulation tool for Hopfield Neural Networks
    %   Detailed explanation goes here
    
    properties
        backupWeights    = [];
        weightMatrix     = [];
        storedPatterns   = [];
        currentState     = [];
        connectionMatrix = [];
        
        % Parameters for delta learning
        symmetricDelta = 0;
        deltaLearningRate = 1;
        deltaLearningThreshold = 0;
        
        unitModel = 'S';
        clipMode  = 'noclip';
        clipValue = 1;
        updateMode = 'async';
        activityNormalization = 'on';
        updateDynamics = 'deterministic';
        learningRule = 'Hebb'; % Can be 'Hebb', 'Storkey', or 'Delta'
        
        beta = 1.0;
        gamma = 1.0;
        networkSize = 0;
        unitThreshold = 0;
        unitValues = [-1 1];
        
        maxIterations = 100;
        maxDeltaIterations = 100;   % Maximum number of times a delta update is performed
        
        inputNoiseMean = 0;
        inputNoiseSd   = 0;
    end
    
    methods
        % Initializes a Hopfield network of a given size
        function obj = Hopfield(networkSize)
            obj.networkSize      = networkSize;
            obj.weightMatrix     = zeros(obj.networkSize);
            obj.unitThreshold    = zeros(obj.networkSize,1);
            obj.currentState     = zeros(obj.networkSize,1);
            obj.connectionMatrix = Connections.GenerateFull(obj.networkSize,0);
        end
        
        % Functions for generating and distorting patterns
        % 1. Generate a single pattern
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
        
        % 2. Generate a matrix of patterns
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
        
        % 3. Distorts a given pattern. Noiselevel should be a value between
        % 0 and 1 that represents the proportion of units that are
        % distorted
        function outputPattern = DistortPattern(obj, inputPattern, noiseLevel)
            flipIndices = randperm(length(inputPattern));
            flipIndices = flipIndices(1:round(noiseLevel*length(inputPattern)));
            
            outputPattern = inputPattern;
            if (obj.unitModel == 'S')
                outputPattern(flipIndices) = -1.*outputPattern(flipIndices);
            elseif (obj.unitModel == 'V')
                outputPattern(flipIndices) = -1.*outputPattern(flipIndices) + 1;
            end
        end
        
        % Stores a single pattern in the network. 
        function StorePattern(obj, nextPattern, C)
            % Input checking and parameter configuration
            if nargin == 2
                C = 1/length(nextPattern);
            end
            
            if obj.networkSize ~= length(nextPattern)
                error('Pattern size not consistent with current weight matrix');
            end
            
            if size(nextPattern,1) == 1
                nextPattern = nextPattern';
            end
            
            % Store the pattern
            obj.storedPatterns = [obj.storedPatterns; nextPattern'];         
            
            if strcmp(obj.activityNormalization,'on')
                nextPattern = nextPattern-mean(nextPattern);
            end
            
            switch obj.learningRule
                case 'Hebb'
                    obj.weightMatrix = obj.gamma.*obj.weightMatrix + C.*obj.connectionMatrix.*(nextPattern*nextPattern');
                case 'Storkey'
                    obj.PerformStorkeyUpdate(nextPattern,C);
                case 'Delta'
                    obj.LearnDeltaPatterns(obj.storedPatterns);
            end
            
            % Post-processing of the weight matrix
            if strcmp(obj.clipMode,'clip')
                obj.weightMatrix(obj.weightMatrix(:) < -obj.clipValue) = -obj.clipValue;
                obj.weightMatrix(obj.weightMatrix(:) > obj.clipValue) = obj.clipValue;
            end           
        end
        
        % Tries to store all the patterns in the patternMatrix in the
        % network
        function StorePatternMatrix(obj,patternMatrix,C)
            if nargin == 2
                C = 1/size(patternMatrix,2);
            end
            
            if strcmp(obj.learningRule,'Delta')
                obj.LearnDeltaPatterns(patternMatrix);
            else
                for i = 1:size(patternMatrix,1)
                    obj.StorePattern(patternMatrix(i,:),C);
                end
            end
        end
        
        % Performs a single adjustment to the weight matrix using the
        % perceptron learning rule for the given input pattern. Returns the
        % number of units that were updated.
        function trainingError = PerformDeltaUpdate(obj, inputPattern, learningRate, learningThreshold)
            if nargin <4
                learningThreshold = obj.deltaLearningThreshold;
            end
            if nargin < 3
                learningRate = obj.deltaLearningRate;
            end
            
            if obj.networkSize ~= length(inputPattern)
                error('Pattern size not consistent with current weight matrix');
            end
            
            % Make sure that we have a column vector
            if size(inputPattern,1) == 1 && size(inputPattern,2) > 1
                inputPattern  = inputPattern';
            end
            
            % Adjust units that fall below the learning threshold
            unitInput = obj.weightMatrix*inputPattern;
            needsUpdating = unitInput.*inputPattern <= learningThreshold;
            adjustUnit = needsUpdating.*inputPattern;
            obj.weightMatrix = obj.weightMatrix + learningRate.*obj.connectionMatrix.*(adjustUnit*inputPattern');
                  
            if obj.symmetricDelta == 1
                obj.weightMatrix = triu(obj.weightMatrix) + triu(obj.weightMatrix,1)';
            end
            
            trainingError = sum(needsUpdating);
        end
        
        % Add patterns from the input matrix using the delta learning rule
        function trainingError = LearnDeltaPatterns(obj, patternMatrix, learningRate, learningThreshold)
            if nargin <4
                learningThreshold = obj.deltaLearningThreshold;
            end
            if nargin < 3
                learningRate = obj.deltaLearningRate;
            end
            obj.storedPatterns = patternMatrix;
            
            currentIteration = 1;
            nPatterns = size(patternMatrix,1);
            
            patternsLearned = 0;
            trainingError = zeros(1,obj.maxDeltaIterations);
            while patternsLearned == 0
                % Iterate over each pattern
                err = zeros(1,nPatterns);
                for trainingIndex = 1:nPatterns
                    err(trainingIndex) = obj.PerformDeltaUpdate(patternMatrix(trainingIndex,:),learningRate, learningThreshold);
                end
                
                % Check if training is complete
                trainingError(currentIteration) = sum(err)/nPatterns;
                if trainingError(currentIteration) == 0
                    patternsLearned = 1;
                else
                    currentIteration = currentIteration + 1;
                    if currentIteration > (obj.maxDeltaIterations)
                        display('Warning! Maximum training iterations for delta learning rule exceeded!');
                        break;
                    end
                end
            end
        end
        
        function PerformStorkeyUpdate(obj, pattern, C)
            nUnits = length(pattern);
            if nargin == 2
                C = 1/nUnits;
            end
            
            if size(pattern,1) == 1
                pattern = pattern';
            end
            
            localField = obj.weightMatrix*pattern;
            % Compute Hebbian term
            hebTerm = C.*(pattern*pattern');
            
            % Compuate middle term 
            pe = repmat(pattern,1,nUnits);
            lfe = repmat(localField',nUnits,1);
            st = obj.weightMatrix.*pe;
            midTerm = C.*(pe.*(lfe-st));
            
            % Compute the rightmost term
            pe = repmat(pattern',nUnits,1);
            lfe = repmat(localField,1,nUnits);
            st = obj.weightMatrix.*pe;
            rightTerm = C.*(pe.*(lfe-st));
            
            
            obj.weightMatrix = obj.weightMatrix + hebTerm - midTerm - rightTerm;
            obj.weightMatrix = obj.weightMatrix.*obj.connectionMatrix;
        end
        
        % Functions for manipulating the weight matrix
        function ResetWeightMatrix(obj)
            obj.storedPatterns = [];
            obj.weightMatrix = zeros(obj.networkSize);
        end   
        function result = GetWeightMatrix(obj)
            result = obj.weightMatrix;
        end
        function SetWeightMatrix(obj, weightMatrix)
            obj.weightMatrix = weightMatrix;
        end
        
        % Network configuration functions
        % 1. Set the connectivity matrix for the units
        function SetConnectionMatrix(obj, cm)
            if size(cm) ~= size(obj.weightMatrix)
                error('Connection matrix dimensions do not match the weight matrix dimensions.');
            end
            
            obj.connectionMatrix = cm;
        end
        % 2. Set the unit model ( 'S' : [-1, 1] or 'V' : [0, 1])
        function SetUnitModel(obj,unitModel)
            if unitModel == 'S'
                obj.unitModel = 'S';
                obj.unitValues = [-1 1];
            elseif unitModel == 'V'
                obj.unitModel = 'V';
                obj.unitValues = [0 1];
            end
        end
        % 3. Set gamma. This parameter controls the proportion of the
        % original weight matrix that is retained in an update step
        function SetGamma(obj,g)
            obj.gamma = g;
        end
        % 4. Set the threshold value for each unit. With a single input value
        % each unit gets the same threshold. A vector with the same length
        % as the number of units can also be provided to set an individual
        % threshold for each unit
        function SetThreshold(obj,T)
            if length(T) == 1
                obj.unitThreshold = T.*ones(obj.networkSize,1);
            elseif length(T) == obj.networkSize
                obj.unitThreshold = T;
                if size(obj.unitThreshold,1)==1
                    obj.unitThreshold = obj.unitThreshold';
                end
            else
                error('Length of T needs to be equal to one or to the number of units in the network');
            end
        end
        % 5. Choose between 'sync' and 'async' for synchronous and
        % asynchronous updating
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
        % 6. Controls if binary patterns should have mean activity of zero
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
        % 7. Set if weights should be restricted to a maximal value
        function EnableWeightClipping(obj, clipValue)
            obj.clipMode = 'clip';
            obj.clipValue = clipValue;
        end
        function DisableWeightClipping(obj)
            obj.clipMode = 'noclip';
        end
        % 8. Do not use stochasticity when updating network state
        function UseDeterministicDynamics(obj)
            obj.updateDynamics = 'deterministic';
        end
        % 9. Use stochasticity. Beta corresponds to the inverse temperature
        function UseStochasticDynamics(obj,beta)
            obj.updateDynamics = 'stochastic';
            obj.beta = beta;
        end
        % 10. Gets the current state of the network
        function currentState = GetCurrentState(obj)
            currentState = obj.currentState;
        end
        % 11. Set parameters for gaussian noise that should be added to the
        % network
        function SetInputNoise(obj, mu, sigma)
            obj.inputNoiseMean = mu;
            obj.inputNoiseSd   = sigma;
        end
        % 12. Clamp the state of the model to a specific set of values
        function ClampState(obj,newState)
            if length(newState) ~= obj.networkSize
                error('Number of units does not correspond')
            end
            if size(newState,2) == 1
                newState = newState';
            end
            obj.currentState = newState;
        end
        % 13. Sets the learning rule to be used for adding new patters
        function SetLearningRule(obj, newRule, param1,param2)
            if  strcmpi(newRule,'Hebb')
                obj.learningRule = 'Hebb';
            elseif strcmpi(newRule,'Storkey')
                obj.learningRule = 'Storkey';
            elseif strcmpi(newRule,'Delta')
                if nargin < 4
                    obj.deltaLearningRate = 1/obj.networkSize;
                else
                    obj.deltaLearningRate = param2;
                end
                if nargin < 3
                    obj.deltaLearningThreshold = 0;
                else
                    obj.deltaLearningThreshold = param1;
                end
                
                obj.learningRule = 'Delta';
            else
                obj.learningRule = 'Hebb';
                error('No valid learning rule. Using Hebb rule as default');
            end
                
        end
        % Here we perform the state update of the hopfield network. Updates
        % are performed synchronously (i.e. all units at the same time), or
        % asynchronously (each unit in turn, but in random order)
        function output = UpdateState(obj, initialState)
            threshold = obj.unitThreshold;
            obj.ClampState(initialState);
            
            switch obj.updateMode
                case 'sync'
                    inputPotential = obj.weightMatrix*obj.currentState';
                    inputPotential = inputPotential + obj.inputNoiseMean + obj.inputNoiseSd.*randn(size(inputPotential));
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
                    updateOrder = randperm(size(obj.weightMatrix,1));
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
            
            inputNoise = obj.inputNoiseMean + obj.inputNoiseSd*rand();
            inputPotential = obj.weightMatrix(unitIndex,:)*obj.currentState' + inputNoise; 
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
                
            if strcmp(obj.updateMode, 'sync')
                notConverged = 1;
                while notConverged == 1
                    if it > obj.maxIterations
                        display('Warning: maximum number of iterations exceeded');
                        break
                    end
                    
                    twoBackState = initialState;
                    initialState = finalState;
                    finalState = obj.UpdateState(initialState);
                    it = it + 1;
                    
                    if sum(finalState~= initialState) == 0
                        notConverged = 0;
                    elseif sum(finalState ~= twoBackState) == 0
                        display('Update cycle detected.');
                        notConverged = 0;
                    end
                end
                
            else
                while sum(initialState ~=  finalState) > 0
                    if it > obj.maxIterations
                        display('Warning: maximum number of iterations exceeded');
                        break
                    end

                    initialState = finalState;
                    finalState = obj.UpdateState(initialState);
                    it = it + 1;
                end
            end
        end
        
        % Returns the energy associated with a specific network state. The
        % function does not take into account if the given state is stable
        % or not.
        function energy = GetEnergy(obj, networkState)
            if (size(networkState,2) > 1)
                networkState = networkState';
            end
            
            energy = -sum(sum(obj.weightMatrix.*(networkState*networkState')));
        end
        
        % Returns the energy associated with each unit in the following
        % state. The function does not take into account if the given state
        % is stable or not. Unit energy is defined as the activity of the
        % unit multiplied by its net input
        function unitEnergy = GetUnitEnergy(obj,networkState)
            if size(networkState,1) == 1
                networkState = networkState';
            end
            
            netInput = obj.weightMatrix*networkState;
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
            obj.weightMatrix = obj.weightMatrix + deltaW;
            for i = 1:size(obj.weightMatrix,1)
                obj.weightMatrix(i,i) = 0;
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
                       
            potentialValues = zeros(1,size(obj.weightMatrix,1)*nPatterns);
            activityValues = zeros(1,size(obj.weightMatrix,1)*nPatterns);
            
            for patternIndex = 1:nPatterns
                nextPattern = obj.storedPatterns(patternIndex,:);
                inputPotential = obj.weightMatrix*nextPattern';                
                potentialValues( (1:size(obj.weightMatrix,1))+((patternIndex-1) * size(obj.weightMatrix,1))) = inputPotential;
                activityValues((1:size(obj.weightMatrix,1))+((patternIndex-1) * size(obj.weightMatrix,1))) = nextPattern;
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
        
        % Returns the overlap for each pattern in patternMatrix when the
        % network is initialized with a noisy version of each pattern and
        % allowed to settle in a stable state
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
        
        % Checks if the patterns given in testPattern are all stable states
        % of the network. That is, after one update iteration the state of
        % the network should not change
        function isStable = IsStablePattern(obj,testPattern)
            nPatterns = size(testPattern,1);
            isStable = zeros(1,nPatterns);
            
            for i = 1:nPatterns
                input = testPattern(i,:);
                output = obj.UpdateState(input);
            
                if sum(output==input) == obj.networkSize
                    isStable(i) = 1;
                else
                    isStable(i) = 0;
                end
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
        
        % Estimates the basin of attraction for each of the patterns
        % Keeps a portion of the pattern fixed and randomizes the other
        % part. If all samples converge we say this is the radius 
        % Returns how many units need to be the same before all samples
        % converge to the original pattern
        function currentDistance = HammingRadius(hopfieldModel, pattern, nSamples)
            currentDistance = length(pattern);
            radiusFound = 0;
            
            while radiusFound == 0
                
                e = zeros(1,nSamples);
                for i = 1:nSamples
                    distortSample = hopfieldModel.DistortPattern(pattern, currentDistance/length(pattern));
                    output = hopfieldModel.Converge(distortSample);
                    e(i) = sum(output ~=pattern);
                    
                    if e(i) > 0
                        break
                    end
                end
                
                if sum(e) == 0
                    radiusFound = 1;
                else
                    currentDistance = currentDistance - 1;
                    if currentDistance == 0
                        display('Warning! Input pattern is not a stable state of the network');
                        radiusFound = 1;
                    end
                end
            end
        end
        
    end
end

