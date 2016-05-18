# Concept learning

Here we applied the procedure described in Gernuschi-Frias, B., & Segura, E. C. (1993). Concept learning in Hopfield associative memories trained with noisy examples using the Hebb rule. Proceedings of 1993 international Joint Conference on Neural Networks. They demonstrated that if a Hopfield network is presented with noisy versions of a prototypical pattern, the prototypical pattern will become a stable state of the network. The simulation shows that at the moment the individual exemplars are no longer stable states of the network, a new stable state emerges that converges to the prototypical pattern. In the figure we also show the results of a useful function of the Hopfield class. This function probes the network with a number of random patterns. This number is specified as an input argument. The function will then return the stable states of the network together with the frequency with which they occured.

###### Probing spurious states
```matlab
[stableStats, stateFrequency] = myHopfield.GetSpuriousStates(numberOfProbes)
```


![Concept learning](/ConceptLearning/Concept_learning.jpg?raw=true)
