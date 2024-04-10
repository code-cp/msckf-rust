import numpy as np 

class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)

def generate_initial_guess(T_c1_c2, z1, z2):
    """
    Compute the initial guess of the feature's 3d position using 
    only two views.

    Arguments:
        T_c1_c2: A rigid body transformation taking a vector from c2 frame 
            to c1 frame. (Isometry3d)
        z1: feature observation in c1 frame. (vec2)
        z2: feature observation in c2 frame. (vec2)

    Returns:
        p: Computed feature position in c1 frame. (vec3)
    """
    # Construct a least square problem to solve the depth.
    m = T_c1_c2.R @ np.array([[*z1, 1.0]]).T
    a = np.squeeze(m[:2]) - z2*m[2]                   # vec2
    b = z2*T_c1_c2.t[2] - T_c1_c2.t[:2]   # vec2

    # Solve for the depth.
    depth = (a.T @ b) / (a.T @ a)
    
    p = np.array([*z1, 1.0]) * depth
    return p

def main(): 
    z1 = np.array([-0.5673448830708987, -0.25667182743780836])
    z2 = np.array([-0.5861749842632805, -0.24109950365363333])
    T_c1_c2 = [            0.999997256477881,
            0.002312067192424,
            0.000376008102415,
            -0.110073808127187,
            -0.002317135723281,
            0.999898048506644,
            0.014089835846648,
            0.000399121547014,
            -0.000343393120525,
            -0.014090668452714,
            0.999900662637729,
            -0.000853702503357,
            0.0,
            0.0,
            0.0,
            1.000000000000000,]
    T_c1_c2 = np.array(T_c1_c2).reshape(4, 4)
    T_c1_c2 = Isometry3d(T_c1_c2[:3, :3], T_c1_c2[:3, 3])

    # in camera 1 frame 
    initial_guess = generate_initial_guess(T_c1_c2, z1, z2)
    initial_guess = initial_guess.reshape(3,1)
    print(f"{initial_guess=}")

    orientation = np.array([[-0.7383999343242164, 0.6737005274982417, -0.01348599876510516], [0.2556494022671243, 0.2615778114207433, -0.9303274490938409], [-0.6234562190724243, -0.6906470746696086, -0.36551018063894]]) 
    position = np.array([[1.7123909322950426, 1.6380334630257034, 1.2853337868985146]]).T 
    solution = orientation @ initial_guess + position
    print(f"{solution=}")                                                                                                                                                                                                                                                                                            

    # initial_guess=array([[-3.01331885],
    #     [-1.36325202],
    #     [ 5.3112647 ]])
    # solution=array([[ 2.94737406],
    #     [-4.43013151],
    #     [ 2.16421086]])

    # output from rust 
    #   ┌                    ┐
    #   │ 1.5655546008018995 │
    #   │ 1.2113014235016237 │
    #   │  1.236626584089743 │
    #   └                    ┘

if __name__ == "__main__": 
    main()