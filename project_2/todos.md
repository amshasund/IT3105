## TODOs
- [ ] Not alout to overwrite another players piece on the board
## Questions to studass
- [ ] Does reformat board work with np.where(Piece.player == 1) ?

- [x] SparseCategorialCrossentropy or CategoricalCrossentropy? Should we use one-hot encoding?
- [x] How to know when game is over? A*? Keep track of neighbours?

- [x] Can we use same initializer for weights and biases in neural net
- [x] What values should we use in the initializer
- [x] Should we initialize weights and biases in input/output layer as well?
- [x] Should we use MeanSquaredError as loss function? -> cross-entropy

## Tips from studass
- State: add one for which player makes the move
- Loss: use cross-entropy - this is also stated in the project paper