#ifndef _SGN_H_
#define _SGN_H_


template <typename T> 
T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


#endif
