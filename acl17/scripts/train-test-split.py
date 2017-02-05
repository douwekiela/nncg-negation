import json as js
import numpy as np
from argparse import ArgumentParser

def main(data_file, train_file, test_file, dev_file, num_noise):
    data = js.load(open(data_file, "r"))
    train_data = []
    test_data = []
    dev_data = []
    
    np.random.seed(100)
    test_indices = np.random.choice(a=range(len(data)),
                                    size=int(0.2*len(data)),
                                    replace=False)

    def distinct_choice(n, k):
        num = 0
        indices = []
        while num<k:
            ind = np.random.randint(low=0, high=len(data), size=1)
            if not (ind in [n] + indices):
                indices.append(ind[0])
                num +=1
        print "indices: ", indices
        return indices
            
    dev_ind = int(0.25*len(test_indices))
    for j, i in enumerate(test_indices):
        example = data[i]
        noise_inds = distinct_choice(i, num_noise)
        noise = []
        for l in noise_inds:
            noise.append(data[l]["output"])
        example["noise"] = noise
        if j < dev_ind:
            dev_data.append(example)
        else:
            test_data.append(example)
        

    train_data = [ex for i,ex in enumerate(data) if not (i in test_indices)]

    js.dump(train_data, open(train_file, "w"))
    js.dump(test_data, open(test_file, "w"))
    js.dump(dev_data, open(dev_file, "w"))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-df", action="store", dest="data_file")
    parser.add_argument("-dvf", action="store", dest="dev_file")
    parser.add_argument("-trf", action="store", dest="train_file")
    parser.add_argument("-tef", action="store", dest="test_file")
    parser.add_argument("-numn", action="store", dest="num_noise",
                        type = int)
    arg = parser.parse_args()
    main(arg.data_file, arg.train_file, arg.test_file, arg.dev_file, arg.num_noise)
