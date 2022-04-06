## Beyonce 4x4 5000 games
learning_rate = 0.005          
hidden_layers = [10, 30, 30, 10] 
activation_function = ["ReLU", "ReLU","ReLU", "ReLU"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256   
batch_size = 256 

## Ada Lovelace 4x4 5000 games
learning_rate = 0.001          
hidden_layers = [10, 30, 50, 30, 10] 
activation_function = ["ReLU", "ReLU","ReLU", "ReLU", "ReLU"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256   
batch_size = 256 

## Iron Man 5x5 5000 games
learning_rate = 0.005       
hidden_layers = [10, 30, 50, 30, 10]  
activation_function = ["ReLU", "ReLU","ReLU", "ReLU", "ReLU"]
optimizer = "sgd"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256     
batch_size = 256 

## Britney 5x5 5000 games
learning_rate = 0.008       
hidden_layers = [10, 20, 20, 10]  
activation_function = ["sigmoid","sigmoid", "sigmoid", "sigmoid"]
optimizer = "sgd"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256     
batch_size = 256 

## Anna Delvi 5x5 1500 games
learning_rate = 0.007          
hidden_layers = [10, 40, 40, 10] 
activation_function = ["swish", "swish", "swish," "swish"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256     
batch_size = 256  

## Hermione 5x5 1500 games
learning_rate = 0.001  
hidden_layers = [25, 50, 50, 25]  
activation_function = ["hard_sigmoid", "hard_sigmoid", "hard_sigmoid", "hard_sigmoid"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256     
batch_size = 256 

## Luna Lovegood 4x4 1500 games
learning_rate = 0.005        
hidden_layers = [16, 32, 32, 16] 
activation_function = ["hard_sigmoid", "hard_sigmoid", "hard_sigmoid", "hard_sigmoid"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2
k = 256     
batch_size = 256 
### ------------ Better models below -----------------
## Wonder Woman 3x3 500 games, 500 MCTS
learning_rate = 0.01         
hidden_layers = [256, 256]  
activation_function = ["ReLU", "ReLU"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2  
k = 256    
batch_size = 256      

## Captain Marvel 4x4 500 games, 500 MCTS
learning_rate = 0.005         
hidden_layers = [256, 256, 128]  
activation_function = ["ReLU", "ReLU", "ReLU]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 2  
k = 256    
batch_size = 256

## Kim Possible 3x3, 4x4 & 5x5 : 500 games, 500 MCTS
learning_rate = 0.01         
hidden_layers = [256, 256]  
activation_function = ["ReLU", "ReLU"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 5  
k = 256    
batch_size = 256  

## Little Possible 3x3, 4x4, 5x5 : 500 games, 500 MCTS
learning_rate = 0.0001         
hidden_layers = [256, 256]  
activation_function = ["ReLU", "ReLU"]
optimizer = "adam"  
num_cached = 6
train_interval = 5 
epochs = 5 or 10 
k = 256    
batch_size = 256  