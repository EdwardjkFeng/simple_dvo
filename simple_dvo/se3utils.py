import numpy as np

# https://pypi.org/project/sophuspy/
import sophus 

def se3_to_Rt(xi: np.ndarray,):
    if not isinstance(xi, np.ndarray):
        raise TypeError(
            "xi must be of type np.ndarray. Got {}.".format(type(xi))
        )

    if not len(xi) == 6:
        raise ValueError(
            "xi must have a shape of (6,). Got {}.".format(xi.shape)
        )
    T = sophus.SE3.exp(xi).matrix()
    R = T[:3, :3]
    t = T[:3, 3]

    return R, t


def Rt_to_se3(R: np.ndarray, t: np.ndarray):
    if not isinstance(R, np.ndarray):
        raise TypeError("R must be of type np.ndarray. Got{}.".format(type(R)))
    if not isinstance(t, np.ndarray):
        raise TypeError("t must be of type np.ndarray. Got{}.".format(type(t)))
    
    if not R.shape == (3, 3):
        raise ValueError(
            "R must have a shape of (3, 3). Got {}.".format(R.shape)
        )
    if not len(t) == 3:
        raise ValueError(
            "t must have a shape of (3). Got {}.".format(t.shape)
        )

    T = sophus.SE3(R, t)
    xi = sophus.SE3.log(T)

    return xi



if __name__ == '__main__':
    xi = np.ones(6)
    R, t = se3_to_Rt(xi)
    print(R, t)

    xi_pi = Rt_to_se3(R, t.squeeze())
    print(xi_pi)
