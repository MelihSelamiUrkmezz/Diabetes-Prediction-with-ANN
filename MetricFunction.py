import numpy as np;

class Loss_CategoricalCrossentropy:
    
    def __init__(self) -> None:
        pass
    
    def forward(self, y_pred: np.ndarray[any, np.dtype[np.float64]] , y_true: np.ndarray) -> np.ndarray[any]:
        samples = len(y_pred);
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7); # for escape infite value forx log()0

        if len(y_true.shape) == 1: # for just index [0, 2]
            correct_confidences = y_pred_clipped[range(samples), y_true];
        
        elif len(y_true.shape) == 2: # for One Hot Encoding [[1, 0, 0], [0, 0, 1]]
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1);

        negative_log_likelihoods = -np.log(correct_confidences);

        return negative_log_likelihoods;

class Loss(Loss_CategoricalCrossentropy):
    sample_loss : np.ndarray[any] = None;
    data_loss : float = None;

    def __init__(self) -> None:
        super().__init__()
    
    def calculate(self, output:np.ndarray[any, np.dtype[np.float64]], y:np.ndarray) -> float:
        self.sample_loss = self.forward(output, y);
        self.data_loss = np.mean(self.sample_loss);
        return self.data_loss;
    
    def to_string(self) -> None:
        print('sample_loss', self.sample_loss);
        print('data_loss', self.data_loss);  


class AccuracyCalculator:

    def __init__(self) -> None:
        pass

    def calculate(self, output:np.ndarray[any, np.dtype[np.float64]], y: np.ndarray): # just support shape 1 for y
        predictions = np.argmax(output, axis=1) # find max for each row
        accuracy = np.mean(predictions==y) # is equals with labels ? 1 : 0 then get mean
        return accuracy;