from .transforms import *

# here we define supported Discrete Groups

class DiscreteGroup:
    transforms = []
    reprs = [1]
    
    def __init__(self, **kwargs):
        pass
        
    def __mul__(self, **kwargs):
        pass
    
    def __repr__(self, **kwargs):
        pass
    
    def __int__(self, **kwargs):
        pass     
    
    def inv(self):
        pass
    
    def __len__(self, **kwargs):
        pass

class Z2(DiscreteGroup):
    transforms = [identity, hflip]
    reprs = [1, 2]
    
    def __init__(self, s=0):
        self.s = s
        
    def __mul__(self, g):
        return Z2((self.s + g.s) % 2)
    
    def __repr__(self):
        return 'e' if self.s == 0 else 's'
    
    def __int__(self):
        return self.s 
    
    def inv(self):
        return Z2(self.s)
    
    def __len__(self):
        return 2
    
    
class P4(DiscreteGroup):
    transforms = [identity, rotate90, rotate180, rotate270]
    reprs = [1, 4]
    
    def __init__(self, r=0):
        self.r = r
        
    def __mul__(self, g):
        return P4((self.r + g.r) % 4)
    
    def __repr__(self):
        return f"r^{self.r}"
    
    def __int__(self):
        return self.r 
    
    def inv(self):
        return P4((4 - self.r) % 4)
    
    def __len__(self):
        return 4
    
    
class P4m(DiscreteGroup):
    transforms = [
        identity, rotate90, rotate180, rotate270,
        hflip, rotate90_hflip, rotate180_hflip, rotate270_hflip
    ]
    reprs = [1, 8]
    
    def __init__(self, r=0, s=0):
        self.r = r
        self.s = s
        
    def __mul__(self, g):
        s_new = (self.s + g.s) % 2
        # commutation relationship
        r_new = (self.r - g.r) % 4 if self.s > 0 else (self.r + g.r) % 4
        return P4m(r_new, s_new)
    
    def __repr__(self):
        if self.r == 0 and self.s == 0:
            return 'e'
        else:
            str_repr = ''
            if self.r > 0:
                str_repr += f"r^{self.r} "
            if self.s > 0:
                str_repr += 's'
            return str_repr
    
    def __int__(self):
        return 4 * self.s + self.r
    
    def inv(self):
        if self.s == 0:
            return P4m((4 - self.r) % 4, 0)
        else:
            return P4m(self.r, self.s)
    
    def __len__(self):
        return 8
