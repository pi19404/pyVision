#!/usr/bin/env sage -python
from pyCommon import *
import pywt
from matplotlib.pyplot import plot, show, figure, title
from scipy import interpolate
from tabulate import tabulate
from scipy import integrate



def piecewise(list_of_pairs, var=None):
    """
    Returns a piecewise function from a list of (interval, function)
    pairs.

    ``list_of_pairs`` is a list of pairs (I, fcn), where
    fcn is a Sage function (such as a polynomial over RR, or functions
    using the lambda notation), and I is an interval such as I = (1,3).
    Two consecutive intervals must share a common endpoint.

    If the optional ``var`` is specified, then any symbolic expressions
    in the list will be converted to symbolic functions using
    ``fcn.function(var)``.  (This says which variable is considered to
    be "piecewise".)

    We assume that these definitions are consistent (ie, no checking is
    done).

    EXAMPLES::

        sage: f1(x) = -1
        sage: f2(x) = 2
        sage: f = Piecewise([[(0,pi/2),f1],[(pi/2,pi),f2]])
        sage: f(1)
        -1
        sage: f(3)
        2
        sage: f = Piecewise([[(0,1),x], [(1,2),x^2]], x); f
        Piecewise defined function with 2 parts, [[(0, 1), x |--> x], [(1, 2), x |--> x^2]]
        sage: f(0.9)
        0.900000000000000
        sage: f(1.1)
        1.21000000000000
    """
    return PiecewisePolynomial(list_of_pairs, var=var)
    
Piecewise=piecewise    
    
class PiecewisePolynomial:
    """
    Returns a piecewise function from a list of (interval, function)
    pairs.

    EXAMPLES::

        sage: f1(x) = -1
        sage: f2(x) = 2
        sage: f = Piecewise([[(0,pi/2),f1],[(pi/2,pi),f2]])
        sage: f(1)
        -1
        sage: f(3)
        2
    """
    def __init__(self, list_of_pairs, var=None):
        r"""
        ``list_of_pairs`` is a list of pairs (I, fcn), where
        fcn is a Sage function (such as a polynomial over RR, or functions
        using the lambda notation), and I is an interval such as I = (1,3).
        Two consecutive intervals must share a common endpoint.

        If the optional ``var`` is specified, then any symbolic
        expressions in the list will be converted to symbolic
        functions using ``fcn.function(var)``.  (This says which
        variable is considered to be "piecewise".)

        We assume that these definitions are consistent (ie, no checking is
        done).

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1 - x
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2]])
            sage: f.list()
            [[(0, 1), x |--> 1], [(1, 2), x |--> -x + 1]]
            sage: f.length()
            2
        """
        self._length = len(list_of_pairs)
        self._intervals = [x[0] for x in list_of_pairs]
        functions = [x[1] for x in list_of_pairs]
        if var is not None:
            for i in range(len(functions)):
                if is_Expression(functions[i]):
                    functions[i] = functions[i].function(var)
        self._functions = functions
        # We regenerate self._list in case self._functions was modified
        # above.  This also protects us in case somebody mutates a list
        # after they use it as an argument to piecewise().
        self._list = [[self._intervals[i], self._functions[i]] for i in range(self._length)]    
        
        
        self.vfunc = np.vectorize(self.run,otypes=[np.float])

    def list(self):
        """
        Returns the pieces of this function as a list of functions.

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1 - x
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2]])
            sage: f.list()
            [[(0, 1), x |--> 1], [(1, 2), x |--> -x + 1]]
        """
        return self._list
        
        
    def run(self,x0):
        n = self.length()
        endpts = self.end_points()
        
        for i in range(1,n):
            if x0 == endpts[i]:
                return (self.functions()[i-1](x0) + self.functions()[i](x0))/2
        if x0 == endpts[0]:
            return self.functions()[0](x0)
        if x0 == endpts[n]:
            return self.functions()[n-1](x0)
        for i in range(n):
            if endpts[i] < x0 < endpts[i+1]:
               
                return self.functions()[i](x0)
        raise ValueError,"Value not defined outside of domain."        
        
    def __repr__(self):
        """
        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1 - x
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2]]); f
            Piecewise defined function with 2 parts, [[(0, 1), x |--> 1], [(1, 2), x |--> -x + 1]]
        """
        return 'Piecewise defined function with %s parts, %s'%(
            self.length(),self.list())
            
            
    def __call__(self,x0):
        """
        Evaluates self at x0. Returns the average value of the jump if x0
        is an interior endpoint of one of the intervals of self and the
        usual value otherwise.

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1-x
            sage: f3(x) = exp(x)
            sage: f4(x) = sin(2*x)
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2],[(2,3),f3],[(3,10),f4]])
            sage: f(0.5)
            1
            sage: f(5/2)
            e^(5/2)
            sage: f(5/2).n()
            12.1824939607035
            sage: f(1)
            1/2
        """
        
        return self.vfunc(x0)



    def length(self):
        """
        Returns the number of pieces of this function.

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1 - x
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2]])
            sage: f.length()
            2
        """
        return self._length        
        
        
    def intervals(self):
        """
        A piecewise non-polynomial example.

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1-x
            sage: f3(x) = exp(x)
            sage: f4(x) = sin(2*x)
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2],[(2,3),f3],[(3,10),f4]])
            sage: f.intervals()
            [(0, 1), (1, 2), (2, 3), (3, 10)]
        """
        return self._intervals


    def functions(self):
        """
        Returns the list of functions (the "pieces").

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1-x
            sage: f3(x) = exp(x)
            sage: f4(x) = sin(2*x)
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2],[(2,3),f3],[(3,10),f4]])
            sage: f.functions()
            [x |--> 1, x |--> -x + 1, x |--> e^x, x |--> sin(2*x)]
        """
        return self._functions


    def domain(self):
        """
        Returns the domain of the function.

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1-x
            sage: f3(x) = exp(x)
            sage: f4(x) = sin(2*x)
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2],[(2,3),f3],[(3,10),f4]])
            sage: f.domain()
            (0, 10)
        """
        endpoints = sum(self.intervals(), ())
        return (min(endpoints), max(endpoints))


    def end_points(self):
        """
        Returns a list of all interval endpoints for this function.

        EXAMPLES::

            sage: f1(x) = 1
            sage: f2(x) = 1-x
            sage: f3(x) = x^2-5
            sage: f = Piecewise([[(0,1),f1],[(1,2),f2],[(2,3),f3]])
            sage: f.end_points()
            [0, 1, 2, 3]
        """
        intervals = self.intervals()
        return [ intervals[0][0] ] + [b for a,b in intervals]        