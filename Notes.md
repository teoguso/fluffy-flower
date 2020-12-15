### Unexplored possibilities

- One-hot representation for restaurants and cities 
  (memory issues so I gave up after a while);

- Proper hyperparameter optimization
  (only tried a few simple values and not all parameters);
- Usage of more data instead of removing part of
  the majority class for balancing (using techniques
  for imbalanced learning);
  
- Analysis of feature importance;

- Proper use of holidays
  (if actual country was known);
  
- Creation of proper data transformers independent of
 training data (so that they could be reused at test time);
  
- Fix random seed everywhere for reproducibility;

- Evaluation of a common model using the two models developed
  (Majority classifier with probability);

- Tests
