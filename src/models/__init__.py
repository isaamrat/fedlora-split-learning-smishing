# models — classical deep learning model definitions for FedSmishGuard
#
# Each module exposes:
#   - A full model class (used by train_classical_fed.py)
#   - Client + Server split classes (used by train_classical_split.py)
#
# Imports:
#   from models.textcnn   import TextCNN, TextCNNClient, TextCNNServer
#   from models.lstm      import LSTMClassifier, BiLSTMClassifier,
#                                LSTMClient, LSTMServer, BiLSTMClient, BiLSTMServer
#   from models.gru       import GRUClassifier, BiGRUClassifier,
#                                GRUClient, GRUServer, BiGRUClient, BiGRUServer
#   from models.cnn_lstm  import CNNLSTMClassifier,
#                                CNNLSTMClientA, CNNLSTMServerA,   # split after embedding
#                                CNNLSTMClientB, CNNLSTMServerB    # split after CNN
