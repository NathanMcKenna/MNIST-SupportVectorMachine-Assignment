import numpy as np 

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector w. 
    The vector w should be returned as an Numpy array. 
    """

    w = np.zeros(len(x[0]))
    for coordinate, label, alphaValue in zip(x, y, alpha):
		w += coordinate * label *alphaValue
	
    return w



def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a set of training examples and primal weights, return the indices 
    of all of the support vectors as a set. 
    """

    support = set()
    
    for index,coordinate in enumerate(x):
        svmLine = np.dot(coordinate,w) + b
        if(1-tolerance <= svmLine <= 1+tolerance or -1-tolerance <= svmLine <= -1+tolerance):
            support.add(index)  
        
    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices 
    of all examples with nonzero slack as a set.  
    """

    slack = set()
    index = 0
    for coordinate,label in zip(x,y):
		if(label*(np.dot(coordinate,w)+b) < 1):
			slack.add(index)
		index += 1
    
    return slack


