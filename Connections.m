classdef Connections < handle
    methods (Static)
        % Generates a matrix in which all units are connected to each other
        function cm = GenerateFull(nUnits, self)
            if nargin == 1
                self = 0;
            end
            
            cm = ones(nUnits);
            
            if self == 0
                for i = 1:nUnits
                    cm(i,i) = 0;
                end
            end
        end

        % Generates connectivity where connections between units are random
        % but each unit receives a fixed number k of synaptic inputs
        function cm = GenerateRandom(nUnits, k, sym)
            if nargin == 2
                sym = 0;
            end
            if k >= nUnits
                error('k cannot be larger than the number of units');
            end
            
            cm = zeros(nUnits);
            
            if sym== 0
                for i = 1:nUnits
                    availableUnits = setdiff(1:nUnits, i);
                    selectedUnits = randperm(length(availableUnits));

                    cm(i,selectedUnits(1:k)) = 1;
                end
            else
               % Attempt at an algorithm to produce a matrix where
               % connections between neurons are random, but the overall
               % connection matrix should be symmetric. 
               % DOESN'T WORK!
               validOperations = k.*ones(1,nUnits);
               while sum(validOperations) ~= 0
                   currentMax = max(validOperations);
                   I = find(validOperations == currentMax);
                   I = I(randperm(length(I)));
                   
                   x = I(1);
                   availableConnections = find(validOperations== currentMax);
                   if length(availableConnections) == 1
                       availableConnections = find(validOperations == (currentMax-1));
                   else
                       availableConnections = setdiff(availableConnections,x);
                   end
                   availableConnections = availableConnections(randperm(length(availableConnections)));
                   y = availableConnections(1);
                   
                   validOperations(x) = validOperations(x)-1;
                   validOperations(y) = validOperations(y)-1;
                   
                   cm(x,y) = 1;
                   cm(y,x) = 1;
               end
            end
        end
        
        % Generates a locally connected network
        function cm = GenerateLocal(nUnits, k)
            if mod(k,2) == 1
                k = k + 1;
            end
            
            cm = zeros(nUnits);
            for i = 1:nUnits
                idx = i-1;
                incoming = mod((idx - round(k/2)):(idx+round(k/2)),nUnits) + 1;
                cm(i,incoming) = 1;
                
                cm(i,i) = 0;
            end
        end
        % Starts from a random connected network and rewires a portion of
        % the existing connections
        function cm = GenerateSmallworld(nUnits,k,p)
            if k >= nUnits
                error('k cannot be larger than the number of units');
            end
            
            cm = Connections.GenerateLocal(nUnits,k);
            
            nRewired = floor(p*k);
            
            for i = 1:nUnits
                currentI = find(cm(i,:)==1);
                currentI = currentI(randperm(length(currentI)));
                available = setdiff(1:nUnits,i);
                available = available(randperm(length(available)));
                
                cm(i,currentI(1:nRewired)) = 0;
                cm(i,available(1:nRewired)) = 1;
            end
        end
        
        % Evaluates the symmetry of a connection matrix
        function sigma = EvaluateSymmetry(cm)
            if size(cm,1) ~= size(cm,2)
                error('Input matrix must be square')
            end
            
            num = 0;
            denom = 0;
            for i = 1:size(cm,1)
                for j = 1:size(cm,2)
                    num = num + cm(i,j)*cm(j,i);
                    denom = denom + cm(i,j).^2;
                end
            end
            
            sigma = num/denom;
        end
        
        % Creates a connection matrix for units that are arranged in a 2D
        % lattice. 
        function GenerateLattice(nUnits,k)
        end
        
        
    end
end