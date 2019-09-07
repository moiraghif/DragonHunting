The code is organized as follows:

The First Folder is Bilbo World and it contains the code for the first part of the project, 
consisting in Q-Learning, DQL and DQL with GA:
 - **agents.py**, contains the Agent classes for QLearning, DQL and DQL with GA
 - **CreateBilboWorld.py**, contains the World class which is where the agents learns to play
 - **Bilbo_q_learning.py**, is the training file for Q-Learning
 - **q_test.py**, is the file to test the result of the model created by **Bilbo_q_learning**
 - **Bilbo_deep_feels.py**, is the training file for Deep-Q-Learning
 - **deep_feels.py**, is the file to test the result of the model created by **Bilbo_deep_feels.py**
 - **Bilbo_ga_deep_feels.py**, is the training file for the DQL with Genetic Algorithm and it contains all the function necessary for the GA.
 - **ga_deep_test.py**, is the file to test the result of the model created by **Bilbo_ga_deep_feels.py**
 - The other files such are **renderer.py** (used for the policy plot of Q-Learning), **ga_vs_deep_comparision.py** are not necessary, they are used just for plots.
 - The models are saved in the models folder and the plots are saved the plots folder
 
 The Second Folder is Duel:
 - **agents.py**, contains the agents class
 - **constants.py**, contains the contants such as hyperparameters, obstacle position.
 - **train.py**, is the training file
 - **world.py**, contains the world class
 - **main.py**, which recalls the **gui.py**, is for testing purpose and creates on two separate threads, the real time animation and QLearning tests.
 - **\*.npy** files are the models saved as numpy array
 
 The Third Folder is Duel_2:
 - it's structure is the same as Duel, the graph integration is done in the **world.py** file
 
 The Last Folder is TeamWork, it has a similar structure to the Duel folder, except:
 - the **world.py** is renamed **GridWorld.py**
 - the **main.py** is renamed **test.py** 
 - **stats_test.py** is used to test over a lot of games without animation
 - **graph.py** plots the graph structure
 - **deep_train.py** is a file for DeepLearning model, which was implemented but requires a lot of training, so was not used in the end
 - the models are saved in the models folder
