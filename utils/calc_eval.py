class Calc_eval:
    def __init__(self, **kwargs):
        self.eval_dict = {}
        self.count = 0

        for metric in kwargs.keys():
            self.eval_dict[metric] = 0.0

    def input_data(self, **kwargs):
        for metric, value in kwargs.items():
            self.eval_dict[metric] += value
        self.count+=1
    
    def get_result_data(self):
        for metric in self.eval_dict.keys():
            self.eval_dict[metric] /= self.count
        return self.eval_dict
